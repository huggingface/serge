#!/usr/bin/env bash
# Push an exported ACM certificate to the box and stand up nginx HTTPS in
# front of the app. Run locally; reads .deploy-state.json for the instance.
#
#   CERT_FILE=cert.pem KEY_FILE=key.pem [CHAIN_FILE=chain.pem] ./aws/setup-tls.sh
#
#   CERT_FILE  : leaf certificate PEM (or leaf+chain already combined).
#   KEY_FILE   : the DECRYPTED private key PEM. If you exported via
#                `aws acm export-certificate` (which encrypts the key with a
#                passphrase), decrypt it first, e.g.:
#                  openssl rsa -in enc-key.pem -out key.pem   # prompts passphrase
#   CHAIN_FILE : optional separate chain PEM, appended after the leaf.
#
# DNS already points serge.huggingface.tech at the box and the instance SG
# already admits 80/443 from the VPN (same-SG rule), so this only touches
# the box: cert files + nginx.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.deploy-state.json"

for cmd in aws jq ssh base64 openssl; do
  command -v "$cmd" >/dev/null || { echo "missing dependency: $cmd" >&2; exit 1; }
done

[[ -f "$STATE_FILE" ]] || { echo "no state file at $STATE_FILE — run deploy.sh first." >&2; exit 1; }

CERT_FILE="${CERT_FILE:?set CERT_FILE to the PEM certificate (leaf or leaf+chain)}"
KEY_FILE="${KEY_FILE:?set KEY_FILE to the (decrypted) PEM private key}"
CHAIN_FILE="${CHAIN_FILE:-}"
APP_HOST="${APP_HOST:-serge.huggingface.tech}"
APP_PORT="${APP_PORT:-8080}"

[[ -f "$CERT_FILE" ]] || { echo "CERT_FILE not found: $CERT_FILE" >&2; exit 1; }
[[ -f "$KEY_FILE" ]] || { echo "KEY_FILE not found: $KEY_FILE" >&2; exit 1; }
[[ -z "$CHAIN_FILE" || -f "$CHAIN_FILE" ]] || { echo "CHAIN_FILE not found: $CHAIN_FILE" >&2; exit 1; }

# A passphrase-encrypted key would make nginx prompt at startup (and hang as
# a service). Refuse it with a decrypt hint rather than shipping it.
if grep -q "ENCRYPTED" "$KEY_FILE"; then
  echo "KEY_FILE is passphrase-encrypted. Decrypt it first, e.g.:" >&2
  echo "  openssl pkey -in $KEY_FILE -out key.pem   # prompts for the passphrase" >&2
  echo "then re-run with KEY_FILE=key.pem" >&2
  exit 1
fi

# Make sure the key actually matches the certificate (works for RSA and EC).
cert_pub="$(openssl x509 -in "$CERT_FILE" -noout -pubkey 2>/dev/null || true)"
key_pub="$(openssl pkey -in "$KEY_FILE" -pubout 2>/dev/null || true)"
if [[ -z "$cert_pub" || "$cert_pub" != "$key_pub" ]]; then
  echo "cert/key mismatch: $KEY_FILE is not the private key for $CERT_FILE" >&2
  exit 1
fi

REGION="$(jq -r .region "$STATE_FILE")"
INSTANCE_ID="$(jq -r .instance_id "$STATE_FILE")"
KEY_PEM="$(jq -r .key_file "$STATE_FILE")"
PRIVATE_IP="$(jq -r .private_ip "$STATE_FILE")"

[[ -f "$KEY_PEM" ]] || { echo "missing SSH key: $KEY_PEM" >&2; exit 1; }

# Refresh the cached private IP if AWS moved it (mirrors update.sh).
state="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || true)"
if [[ "$state" != "running" ]]; then
  echo "instance $INSTANCE_ID is not running (state=$state)" >&2
  exit 1
fi
CURRENT_IP="$(aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)"
[[ -n "$CURRENT_IP" && "$CURRENT_IP" != "None" ]] && PRIVATE_IP="$CURRENT_IP"

# Build the fullchain (leaf [+ chain]) and base64 the payloads.
FULLCHAIN_B64="$( { cat "$CERT_FILE"; [[ -n "$CHAIN_FILE" ]] && cat "$CHAIN_FILE"; } | base64 | tr -d '\n')"
KEY_B64="$(base64 < "$KEY_FILE" | tr -d '\n')"
PROVISION_B64="$(base64 < "${SCRIPT_DIR}/provision-tls.sh" | tr -d '\n')"

SSH_OPTS=(
  -i "$KEY_PEM"
  -o StrictHostKeyChecking=accept-new
  -o UserKnownHostsFile="${SCRIPT_DIR}/.known_hosts"
  -o ConnectTimeout=10
)

REMOTE_SCRIPT=$(cat <<EOF
set -euo pipefail

echo "==> installing cert + key into /etc/reviewbot/tls"
sudo install -d -m 0700 -o root -g root /etc/reviewbot/tls
base64 -d <<'CERT' | sudo tee /etc/reviewbot/tls/fullchain.pem >/dev/null
${FULLCHAIN_B64}
CERT
base64 -d <<'KEY' | sudo tee /etc/reviewbot/tls/privkey.pem >/dev/null
${KEY_B64}
KEY
sudo chown root:root /etc/reviewbot/tls/fullchain.pem /etc/reviewbot/tls/privkey.pem
sudo chmod 0600 /etc/reviewbot/tls/fullchain.pem /etc/reviewbot/tls/privkey.pem

echo "==> running provision-tls.sh"
base64 -d <<'PROV' > /tmp/provision-tls.sh
${PROVISION_B64}
PROV
chmod +x /tmp/provision-tls.sh
sudo /tmp/provision-tls.sh "${APP_HOST}" "${APP_PORT}"
rm -f /tmp/provision-tls.sh
EOF
)

echo "==> ssh ec2-user@${PRIVATE_IP}"
ssh "${SSH_OPTS[@]}" "ec2-user@${PRIVATE_IP}" "bash -s" <<<"$REMOTE_SCRIPT"

cat <<EOF

==> done. nginx is terminating HTTPS for https://${APP_HOST} -> 127.0.0.1:${APP_PORT}

Verify over the VPN:
    curl -sSf https://${APP_HOST}/healthz && echo

Reminder: an exported ACM cert is a static copy. When it renews/expires,
re-export and re-run this script — the box won't update itself.
EOF
