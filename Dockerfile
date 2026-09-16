# Production image for the serge web app (reviewbot-web). Mirrors the EC2
# host: python3.11 + bubblewrap (so HELPER_SANDBOX can stay on), the
# package installed into a venv with the [web] extra (FastAPI/uvicorn),
# running uvicorn on $PORT (default 8080) as an unprivileged user. The
# embedded SQLite job store persists on a mounted volume (see chart/).
#
# The sandbox-verification image used for local bwrap testing lives at
# docker/Dockerfile and is unrelated to this one.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged service user, mirroring ec2-user on the real host.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /opt/app
COPY . /opt/app
# The [kubernetes] extra ships the API client for the kubernetes normalize
# backend (TASK_SANDBOX_BACKEND=kubernetes). It's imported lazily, so it adds
# no startup cost when the backend is unused (docker/bwrap deployments).
RUN python -m venv /opt/app/.venv \
    && /opt/app/.venv/bin/pip install --upgrade pip \
    && /opt/app/.venv/bin/pip install -e '.[web,kubernetes]'

# The `relore` client, for the project-history tools (reviewbot/relore_tool.py).
#
# PINNED, and the pin is a contract, not a convenience. relore's client and
# daemon must be the exact same version: a mismatch is refused with 426 rather
# than answered, because an older client would otherwise get a complete-looking
# reply missing whatever it does not know to ask for. So RELORE_REF must name
# the commit the deployed daemon was built from, and bumping one without the
# other is caught loudly at the first call instead of quietly in the answers.
#
# Installed at BUILD time on purpose: the task pod's egress allowlist has no
# PyPI (and no github.com raw access for pip), so this cannot be a runtime
# install. `relore` alone — NOT `relore[server]`, which drags in SQLAlchemy and
# a web server serge has no use for.
# The repo publishes no tags, and its own deployment pins `image.tag:
# sha-<short>`, so this is the same commit spelled the way pip takes it.
# Keep it equal to relore/env/prod.yaml's image.tag in the playbooks repo.
ARG RELORE_REF=e0cbec92155e02b3a3bf495767e5a209b6f6e50c  # relore 0.3.17, the deployed daemon
RUN /opt/app/.venv/bin/pip install --no-cache-dir \
      "relore @ git+https://github.com/huggingface/relore@${RELORE_REF}"

ENV PATH="/opt/app/.venv/bin:${PATH}"
ENV PORT=8080

# Bake the build commit into the image so /version, the JSON `serge` stamp,
# and the web UI footer report exactly what is deployed. CI passes this from
# github.sha; the .git dir isn't shipped, so the app reads it from here.
ARG GIT_SHA=""
ENV SERGE_GIT_SHA=${GIT_SHA}
EXPOSE 8080
USER app
CMD ["reviewbot-web"]
