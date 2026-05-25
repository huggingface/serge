import logging
import time

import jwt
import requests

log = logging.getLogger(__name__)


class AppNotInstalledError(RuntimeError):
    """Raised when ``/repos/{owner}/{repo}/installation`` returns 404 —
    i.e. the GitHub App is not installed on the target repo. The web UI
    catches this and surfaces an actionable hint instead of crashing
    the worker with a raw HTTPError stack."""

    def __init__(self, owner: str, repo: str):
        self.owner = owner
        self.repo = repo
        super().__init__(
            f"The GitHub App is not installed on {owner}/{repo}. "
            f"Install it from the App's settings page and try again."
        )


def app_jwt(app_id: str, private_key: str) -> str:
    now = int(time.time())
    payload = {"iat": now - 60, "exp": now + 9 * 60, "iss": app_id}
    return jwt.encode(payload, private_key, algorithm="RS256")


def installation_token(app_id: str, private_key: str, installation_id: int) -> str:
    token = app_jwt(app_id, private_key)
    r = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["token"]


def installation_id_for_repo(
    app_id: str, private_key: str, owner: str, repo: str
) -> int:
    """Look up the installation id for a repo via the App JWT. Used by
    web mode to mint an installation token without relying on an
    incoming webhook payload to supply the id."""
    token = app_jwt(app_id, private_key)
    r = requests.get(
        f"https://api.github.com/repos/{owner}/{repo}/installation",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        timeout=30,
    )
    if r.status_code == 404:
        raise AppNotInstalledError(owner, repo)
    r.raise_for_status()
    data = r.json()
    iid = data.get("id")
    if not isinstance(iid, int):
        raise RuntimeError(
            f"GitHub returned no installation id for {owner}/{repo}: {data!r}"
        )
    return iid


def user_is_org_member(
    app_id: str, private_key: str, org: str, username: str
) -> bool:
    """Return True if ``username`` belongs to ``org``.

    Order of checks:

    1. ``GET /orgs/{org}/public_members/{username}`` — unauthenticated,
       only sees publicly listed memberships. Covers the common case
       cheaply with no App permissions required.
    2. ``GET /orgs/{org}/installation`` + ``GET /orgs/{org}/members/{username}``
       using the GitHub App's installation token. Sees private members
       too, but needs the App to be installed on the org and to have
       "Organization Members: read" permission.

    The second path is the workaround for SAML-protected orgs where the
    user's OAuth token returns an empty ``/user/orgs`` list because it
    hasn't been SSO-authorized. The App authenticates as itself, so
    user-side SSO restrictions don't apply.

    Returns False on any failure (App not installed, missing permission,
    network issue, etc.) — the caller decides what to do then."""
    try:
        pub_resp = requests.get(
            f"https://api.github.com/orgs/{org}/public_members/{username}",
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        if pub_resp.status_code == 204:
            return True
    except requests.RequestException:
        log.debug("public_members check for %s/%s failed", org, username, exc_info=True)

    try:
        jwt_token = app_jwt(app_id, private_key)
        inst_resp = requests.get(
            f"https://api.github.com/orgs/{org}/installation",
            headers={
                "Authorization": f"Bearer {jwt_token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        if inst_resp.status_code != 200:
            log.info(
                "App installation lookup for org %s returned %d; cannot verify "
                "private membership via App",
                org,
                inst_resp.status_code,
            )
            return False
        iid = inst_resp.json().get("id")
        if not isinstance(iid, int):
            return False
        token = installation_token(app_id, private_key, iid)
        # ``allow_redirects=False`` so we can distinguish a real "is a
        # member" 204 from GitHub's 302-to-public_members fallback.
        # A 302 means our App authenticated successfully but doesn't
        # have permission to read non-public members — i.e. needs the
        # "Organization Members: read" permission added in the App's
        # settings and accepted by the org owner.
        m_resp = requests.get(
            f"https://api.github.com/orgs/{org}/members/{username}",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
            allow_redirects=False,
        )
        if m_resp.status_code == 204:
            return True
        if m_resp.status_code in (302, 403):
            log.warning(
                "App lacks 'Organization Members: read' permission on %s "
                "(status=%d); cannot verify private membership. Grant the "
                "App that permission and re-accept to enable this path.",
                org,
                m_resp.status_code,
            )
        elif m_resp.status_code == 404:
            log.info(
                "User %s is not a member of org %s per the App view",
                username,
                org,
            )
        else:
            log.info(
                "Unexpected status %d from /orgs/%s/members/%s",
                m_resp.status_code,
                org,
                username,
            )
        return False
    except requests.RequestException:
        log.warning("App-based org membership check failed", exc_info=True)
        return False
