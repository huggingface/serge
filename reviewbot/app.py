import hashlib
import hmac
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, abort, jsonify, request

from .config import Config
from .github_auth import installation_token
from .github_client import GitHubClient
from .reviewer import ReviewRequest, run_followup, run_review
from .triggers import build_review_request

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("ai-reviewer")

cfg = Config.from_env(require_app=True)
app = Flask(__name__)

# Bound the number of concurrent reviews so a single chatty collaborator
# (or a deliberately spammy comment loop) can't exhaust the runner or
# blow up the LLM bill. Tune via WEBHOOK_MAX_WORKERS; the default fits a
# small EC2 box.
_MAX_WORKERS = int(os.environ.get("WEBHOOK_MAX_WORKERS", "2"))
_REVIEW_POOL = ThreadPoolExecutor(
    max_workers=_MAX_WORKERS,
    thread_name_prefix="review-worker",
)


def _verify_signature(body: bytes, header: str) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    assert cfg.github_webhook_secret is not None  # enforced by require_app=True
    mac = hmac.new(cfg.github_webhook_secret.encode(), body, hashlib.sha256)
    expected = "sha256=" + mac.hexdigest()
    return hmac.compare_digest(expected, header)


def _review_worker(installation_id: int, req: ReviewRequest) -> None:
    try:
        assert cfg.github_app_id and cfg.github_private_key
        token = installation_token(
            cfg.github_app_id, cfg.github_private_key, installation_id
        )
        gh = GitHubClient(token)
        if req.inline is not None:
            run_followup(cfg, gh, req)
        else:
            run_review(cfg, gh, req)
    except Exception:
        log.exception("review worker crashed for %s/%s#%d", req.owner, req.repo, req.number)


@app.get("/")
def health() -> tuple:
    return jsonify({"status": "ok"}), 200


@app.post("/webhook")
def webhook() -> tuple:
    sig = request.headers.get("X-Hub-Signature-256", "")
    if not _verify_signature(request.get_data(), sig):
        log.warning("rejected webhook with bad signature")
        abort(401)

    event = request.headers.get("X-GitHub-Event", "")
    payload = request.get_json(silent=True) or {}

    if event == "ping":
        return jsonify({"pong": True}), 200

    req = build_review_request(event, payload, cfg.mention_trigger)
    if req is None:
        return jsonify({"skipped": "trigger"}), 204

    installation = payload.get("installation") or {}
    installation_id = installation.get("id")
    if not isinstance(installation_id, int):
        return jsonify({"skipped": "no_installation"}), 204

    _REVIEW_POOL.submit(_review_worker, installation_id, req)

    return jsonify({"status": "accepted"}), 202


def main() -> int:
    """Console entry point — runs Flask's dev server. Production deployments
    should use a real WSGI server, e.g. `gunicorn reviewbot.app:app`."""
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
