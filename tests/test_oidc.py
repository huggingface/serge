"""Tests for GitHub Actions OIDC verification. We generate an RSA keypair,
sign tokens locally, and stub the JWKS client to return our public key so
no network is touched."""

import time
import unittest

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from reviewbot import oidc
from reviewbot.oidc import OIDCError, verify_token

ISSUER = "https://token.actions.githubusercontent.com"
AUDIENCE = "serge"


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    public_pem = (
        key.public_key()
        .public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


class _FakeSigningKey:
    def __init__(self, key):
        self.key = key


class _FakeJWKClient:
    def __init__(self, key):
        self._key = key

    def get_signing_key_from_jwt(self, token):
        return _FakeSigningKey(self._key)


class OIDCTests(unittest.TestCase):
    def setUp(self) -> None:
        self.private_pem, self.public_pem = _keypair()
        # Route the module's JWKS lookup to our in-memory public key.
        self._orig = oidc._jwks_client
        oidc._jwks_client = lambda issuer: _FakeJWKClient(self.public_pem)
        self.addCleanup(setattr, oidc, "_jwks_client", self._orig)

    def _token(self, **overrides):
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": AUDIENCE,
            "iat": now - 10,
            "exp": now + 300,
            "repository": "acme/widgets",
            "actor": "octocat",
            "workflow_ref": "acme/widgets/.github/workflows/fix.yml@refs/heads/main",
        }
        claims.update(overrides)
        return jwt.encode(claims, self.private_pem, algorithm="RS256")

    def test_valid_token(self):
        claims = verify_token(self._token(), issuer=ISSUER, audience=AUDIENCE)
        self.assertEqual(claims.repository, "acme/widgets")
        self.assertEqual(claims.actor, "octocat")

    def test_wrong_audience_rejected(self):
        tok = self._token(aud="someone-else")
        with self.assertRaises(OIDCError):
            verify_token(tok, issuer=ISSUER, audience=AUDIENCE)

    def test_wrong_issuer_rejected(self):
        tok = self._token(iss="https://evil.example")
        with self.assertRaises(OIDCError):
            verify_token(tok, issuer=ISSUER, audience=AUDIENCE)

    def test_expired_rejected(self):
        now = int(time.time())
        tok = self._token(iat=now - 1000, exp=now - 500)
        with self.assertRaises(OIDCError):
            verify_token(tok, issuer=ISSUER, audience=AUDIENCE)

    def test_missing_repository_rejected(self):
        tok = self._token(repository="")
        with self.assertRaises(OIDCError):
            verify_token(tok, issuer=ISSUER, audience=AUDIENCE)

    def test_empty_token_rejected(self):
        with self.assertRaises(OIDCError):
            verify_token("", issuer=ISSUER, audience=AUDIENCE)


if __name__ == "__main__":
    unittest.main()
