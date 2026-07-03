#!/usr/bin/env bash
# nginx reverse proxy on the box terminating HTTPS with an exported ACM
# certificate, forwarding to the reviewbot-web app on 127.0.0.1:<port>.
#
# Runs ON the EC2 host as root (setup-tls.sh ships and invokes it). The cert
# and key must already be in place at /etc/reviewbot/tls/{fullchain,privkey}.pem
# — setup-tls.sh uploads them before calling this. Safe to re-run.
#
#   provision-tls.sh <host> [upstream-port]
#
# DNS for <host> already points straight at this box, so nginx just needs to
# listen on 443; no load balancer and no DNS change are involved.

set -euxo pipefail

APP_HOST="${1:?usage: provision-tls.sh <host> [upstream-port]}"
UPSTREAM_PORT="${2:-8080}"
CERT_DIR="/etc/reviewbot/tls"
NGINX_CONF="/etc/nginx/conf.d/reviewbot.conf"

dnf install -y nginx

if [[ ! -f "$CERT_DIR/fullchain.pem" || ! -f "$CERT_DIR/privkey.pem" ]]; then
  echo "missing $CERT_DIR/fullchain.pem or privkey.pem — upload them first" >&2
  exit 1
fi

cat >"$NGINX_CONF" <<NGINX
server {
    listen 80;
    server_name ${APP_HOST};
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    http2 on;
    server_name ${APP_HOST};

    ssl_certificate     ${CERT_DIR}/fullchain.pem;
    ssl_certificate_key ${CERT_DIR}/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://127.0.0.1:${UPSTREAM_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # The review console streams via Server-Sent Events for minutes at a
        # time. Disable buffering and allow long-lived upstream reads so
        # tokens reach the browser as they're produced.
        proxy_set_header Connection "";
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
NGINX

# If SELinux is enforcing, allow nginx to open the upstream TCP connection.
if command -v getenforce >/dev/null 2>&1 && [[ "$(getenforce)" == "Enforcing" ]]; then
  setsebool -P httpd_can_network_connect 1 || true
fi

systemctl enable --now nginx
nginx -t
systemctl reload nginx
