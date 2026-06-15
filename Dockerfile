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
RUN python -m venv /opt/app/.venv \
    && /opt/app/.venv/bin/pip install --upgrade pip \
    && /opt/app/.venv/bin/pip install -e '.[web]'

ENV PATH="/opt/app/.venv/bin:${PATH}"
ENV PORT=8080
EXPOSE 8080
USER app
CMD ["reviewbot-web"]
