"""GitHub Actions OIDC verification for the write-capable /tasks endpoint.

A calling workflow declares ``permissions: id-token: write`` and mints a
short-lived, GitHub-signed JWT at job time. serge verifies that token
against the issuer's JWKS (signature + ``iss`` / ``aud`` / ``exp``) and
authorizes the task on the token's ``repository`` claim — it will only act
on the repo named in the token. A leaked token is useless within minutes
and only ever scoped to one repo.

Verification runs in serge's main process (which already reaches the
GitHub API and the LLM providers), not the network-isolated sandbox, so
``--unshare-net`` does not block the JWKS fetch.

The JWKS is fetched and cached by PyJWT's :class:`jwt.PyJWKClient`; we keep
one client per issuer so signing keys are reused across requests and
refreshed on rotation.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

import jwt
from jwt import PyJWKClient

log = logging.getLogger(__name__)


class OIDCError(Exception):
    """Raised when an OIDC token fails verification. The message is safe to
    surface to the caller (no secrets), and the route maps it to 401/403."""


@dataclass
class OIDCClaims:
    """The subset of verified claims the tasks flow cares about."""

    repository: str  # "owner/name" — the repo serge is authorized to act on
    actor: str  # the GitHub login that triggered the workflow run
    workflow_ref: str  # e.g. "owner/name/.github/workflows/fix.yml@refs/heads/main"
    raw: dict[str, Any]


# One PyJWKClient per issuer URI. PyJWKClient caches signing keys
# internally; sharing the instance keeps that cache warm across requests.
_jwks_clients: dict[str, PyJWKClient] = {}
_jwks_lock = threading.Lock()


def _jwks_client(issuer: str) -> PyJWKClient:
    jwks_uri = issuer.rstrip("/") + "/.well-known/jwks"
    with _jwks_lock:
        client = _jwks_clients.get(jwks_uri)
        if client is None:
            client = PyJWKClient(jwks_uri)
            _jwks_clients[jwks_uri] = client
        return client


def verify_token(token: str, *, issuer: str, audience: str) -> OIDCClaims:
    """Verify a GitHub Actions OIDC JWT and return its claims.

    Checks the RS256 signature against the issuer's JWKS plus ``iss``,
    ``aud``, and ``exp`` (PyJWT enforces ``exp``/``iat`` by default).
    Raises :class:`OIDCError` on any failure."""
    if not token:
        raise OIDCError("missing bearer token")
    try:
        signing_key = _jwks_client(issuer).get_signing_key_from_jwt(token)
    except Exception as exc:  # PyJWKClientError, network, etc.
        log.warning("OIDC JWKS lookup failed: %s", type(exc).__name__)
        raise OIDCError("could not resolve token signing key") from exc
    try:
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=audience,
            issuer=issuer,
            options={"require": ["exp", "iss", "aud"]},
        )
    except jwt.InvalidTokenError as exc:
        # Includes expired / bad-audience / bad-issuer / bad-signature.
        raise OIDCError(f"invalid OIDC token: {exc}") from exc

    repository = (claims.get("repository") or "").strip()
    if not repository or repository.count("/") != 1:
        raise OIDCError("token has no usable 'repository' claim")
    return OIDCClaims(
        repository=repository,
        actor=claims.get("actor") or "",
        workflow_ref=claims.get("workflow_ref") or "",
        raw=claims,
    )
