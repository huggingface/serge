# Local Linux image for exercising the bubblewrap helper-sandbox on a macOS
# host (bwrap is Linux-only). Mirrors the EC2 host: python3.11 + bubblewrap,
# the package installed into a venv, running as an unprivileged user.
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Unprivileged service user, mirroring ec2-user on the real host.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /opt/app
COPY . /opt/app
RUN python -m venv /opt/app/.venv \
    && /opt/app/.venv/bin/pip install --upgrade pip \
    && /opt/app/.venv/bin/pip install -e '.[web]' pytest

# A "host secret" that lives OUTSIDE any worktree; the sandbox must never
# expose it to a PR-touching subprocess.
RUN install -d -m 0700 /etc/reviewbot \
    && echo 'TOP-SECRET-GITHUB-APP-KEY' > /etc/reviewbot/github-app.pem \
    && chmod 0600 /etc/reviewbot/github-app.pem

ENV PATH="/opt/app/.venv/bin:${PATH}"
USER app
CMD ["python", "/opt/app/docker/verify_sandbox.py"]
