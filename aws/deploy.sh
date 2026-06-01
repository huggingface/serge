#!/usr/bin/env bash
# Provision a single EC2 host running the interactive reviewboard app.

set -euo pipefail

# ---- config (override via env) ----------------------------------------------
STACK_NAME="${STACK_NAME:-ai-reviewer-reviewboard}"
AWS_REGION="${AWS_REGION:-eu-north-1}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.medium}"
VOLUME_SIZE_GB="${VOLUME_SIZE_GB:-20}"
REPO_URL="${REPO_URL:-https://github.com/huggingface/serge.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
# SUBNET_ID is required — pick a private subnet in your target VPC.
SUBNET_ID="${SUBNET_ID:-}"
# Set SG_ID to reuse an existing security group. Defaults to the VPC's
# `default` security group if unset.
SG_ID="${SG_ID:-}"
SERVICE_NAME="${SERVICE_NAME:-reviewbot-web}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.deploy-state.json"
KEY_FILE="${SCRIPT_DIR}/${STACK_NAME}.pem"
APP_ENV_LOCAL_FILE="${APP_ENV_LOCAL_FILE:-${SCRIPT_DIR}/reviewbot-web.env}"
APP_ENV_REMOTE_FILE="/etc/reviewbot/${SERVICE_NAME}.env"
APP_DIR="/opt/app/ai-reviewer"
PEM_REMOTE_FILE="/etc/reviewbot/github-app.pem"

# Set KEY_NAME to reuse an existing key pair; KEY_FILE must match it.
KEY_NAME="${KEY_NAME:-${STACK_NAME}-key}"

# Track whether we created these resources so destroy knows what to clean up.
SG_CREATED_BY_US=false
KEY_CREATED_BY_US=false

# ---- prereqs ----------------------------------------------------------------
for cmd in aws jq base64 openssl; do
  command -v "$cmd" >/dev/null || { echo "missing dependency: $cmd" >&2; exit 1; }
done

if ! aws sts get-caller-identity --region "$AWS_REGION" >/dev/null 2>&1; then
  echo "aws cli not authenticated. run 'aws configure' or export AWS_PROFILE." >&2
  exit 1
fi

if [[ -f "$STATE_FILE" ]]; then
  echo "state file already exists at $STATE_FILE — run destroy.sh first." >&2
  exit 1
fi

if [[ ! -f "$APP_ENV_LOCAL_FILE" ]]; then
  echo "missing env file: $APP_ENV_LOCAL_FILE" >&2
  echo "copy aws/reviewbot-web.env.example to aws/reviewbot-web.env and fill it in." >&2
  exit 1
fi

read_env_value() {
  local key="$1"
  local file="$2"
  local line

  line="$(grep -E "^${key}=" "$file" | tail -n 1 || true)"
  [[ -n "$line" ]] || return 1

  line="${line#*=}"
  line="${line%$'\r'}"
  if [[ "${line:0:1}" == '"' && "${line: -1}" == '"' ]]; then
    line="${line:1:${#line}-2}"
  elif [[ "${line:0:1}" == "'" && "${line: -1}" == "'" ]]; then
    line="${line:1:${#line}-2}"
  fi

  printf '%s' "$line"
}

APP_ENV_STAGED_FILE="$(mktemp)"
cleanup() {
  rm -f "$APP_ENV_STAGED_FILE"
}
trap cleanup EXIT
cp "$APP_ENV_LOCAL_FILE" "$APP_ENV_STAGED_FILE"

PEM_LOCAL_FILE="$(read_env_value GITHUB_PRIVATE_KEY_PATH "$APP_ENV_LOCAL_FILE" || true)"
PEM_B64=""
if [[ -n "$PEM_LOCAL_FILE" ]]; then
  if [[ "$PEM_LOCAL_FILE" == "~/"* ]]; then
    PEM_LOCAL_FILE="${HOME}/${PEM_LOCAL_FILE#~/}"
  elif [[ "$PEM_LOCAL_FILE" != /* ]]; then
    APP_ENV_DIR="$(cd "$(dirname "$APP_ENV_LOCAL_FILE")" && pwd)"
    PEM_LOCAL_FILE="${APP_ENV_DIR}/${PEM_LOCAL_FILE}"
  fi
  if [[ ! -f "$PEM_LOCAL_FILE" ]]; then
    echo "GITHUB_PRIVATE_KEY_PATH does not exist locally: $PEM_LOCAL_FILE" >&2
    exit 1
  fi
  PEM_B64="$(base64 < "$PEM_LOCAL_FILE" | tr -d '\n')"
  awk -v target="$PEM_REMOTE_FILE" '
    /^GITHUB_PRIVATE_KEY_PATH=/ {
      print "GITHUB_PRIVATE_KEY_PATH=" target
      next
    }
    { print }
  ' "$APP_ENV_LOCAL_FILE" > "$APP_ENV_STAGED_FILE"
fi

# WEB_SESSION_SECRET must be a real random value; the app refuses to
# start with an empty / placeholder secret when DEV_NO_AUTH is off, and
# we don't want a fresh deploy to depend on the operator remembering to
# rotate it. If it's still the example placeholder, mint one and persist
# it back to the local env file so re-runs of update.sh keep the same
# value (otherwise sessions would invalidate on every redeploy).
WEB_SESSION_SECRET_VAL="$(read_env_value WEB_SESSION_SECRET "$APP_ENV_STAGED_FILE" || true)"
if [[ -z "$WEB_SESSION_SECRET_VAL" || "$WEB_SESSION_SECRET_VAL" == "replace-me" ]]; then
  GENERATED_SECRET="$(openssl rand -hex 32)"
  echo "==> minting WEB_SESSION_SECRET (was empty/placeholder); writing it back to $APP_ENV_LOCAL_FILE"
  for target in "$APP_ENV_STAGED_FILE" "$APP_ENV_LOCAL_FILE"; do
    if grep -qE '^WEB_SESSION_SECRET=' "$target"; then
      awk -v val="$GENERATED_SECRET" '
        /^WEB_SESSION_SECRET=/ { print "WEB_SESSION_SECRET=" val; next }
        { print }
      ' "$target" > "${target}.tmp"
      mv "${target}.tmp" "$target"
    else
      printf '\nWEB_SESSION_SECRET=%s\n' "$GENERATED_SECRET" >> "$target"
    fi
  done
fi

APP_ENV_B64="$(base64 < "$APP_ENV_STAGED_FILE" | tr -d '\n')"

# ---- discover inputs --------------------------------------------------------
echo "==> looking up latest Amazon Linux 2023 AMI"
AMI_ID="$(aws ssm get-parameter \
  --region "$AWS_REGION" \
  --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameter.Value' --output text)"
echo "    $AMI_ID"

if [[ -z "$SUBNET_ID" ]]; then
  echo "SUBNET_ID is required. Export SUBNET_ID=<subnet-id> for a subnet in $AWS_REGION." >&2
  exit 1
fi

VPC_ID="$(aws ec2 describe-subnets \
  --region "$AWS_REGION" \
  --subnet-ids "$SUBNET_ID" \
  --query 'Subnets[0].VpcId' --output text)"
echo "    vpc $VPC_ID"

# ---- key pair ---------------------------------------------------------------
echo "==> ensuring SSH key pair $KEY_NAME"
if aws ec2 describe-key-pairs --region "$AWS_REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  echo "    key pair already exists in AWS"
  [[ -f "$KEY_FILE" ]] || {
    echo "    but local $KEY_FILE is missing — delete the AWS key or restore the PEM." >&2
    exit 1
  }
else
  aws ec2 create-key-pair \
    --region "$AWS_REGION" \
    --key-name "$KEY_NAME" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  KEY_CREATED_BY_US=true
  echo "    wrote private key to $KEY_FILE"
fi

# ---- security group ---------------------------------------------------------
if [[ -z "$SG_ID" ]]; then
  echo "==> resolving default security group for $VPC_ID"
  SG_ID="$(aws ec2 describe-security-groups \
    --region "$AWS_REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=group-name,Values=default" \
    --query 'SecurityGroups[0].GroupId' --output text)"
  [[ "$SG_ID" != "None" && -n "$SG_ID" ]] \
    || { echo "no default SG found in $VPC_ID" >&2; exit 1; }
  echo "    $SG_ID"
else
  echo "==> using security group $SG_ID"
  aws ec2 describe-security-groups --region "$AWS_REGION" --group-ids "$SG_ID" \
    --query 'SecurityGroups[0].GroupName' --output text >/dev/null \
    || { echo "security group $SG_ID not found in $AWS_REGION" >&2; exit 1; }
fi

# ---- user-data --------------------------------------------------------------
USER_DATA="$(cat <<EOF
#!/bin/bash
set -euxo pipefail

dnf update -y
# bubblewrap sandboxes the PR-touching helper/context subprocesses; with
# HELPER_SANDBOX=require the app refuses to run them if it's missing.
dnf install -y git python3.11 python3.11-pip bubblewrap

install -d -o ec2-user -g ec2-user /opt/app
sudo -u ec2-user git clone --branch ${REPO_BRANCH} ${REPO_URL} ${APP_DIR}
sudo -u ec2-user /usr/bin/python3.11 -m venv ${APP_DIR}/.venv
sudo -u ec2-user ${APP_DIR}/.venv/bin/pip install --upgrade pip
sudo -u ec2-user ${APP_DIR}/.venv/bin/pip install -e '${APP_DIR}[web]'

install -d -m 0750 -o root -g ec2-user /etc/reviewbot
if [[ -n "${PEM_B64}" ]]; then
cat <<'PEMFILE' | base64 -d > ${PEM_REMOTE_FILE}
${PEM_B64}
PEMFILE
# The service runs as ec2-user, so the PEM is owned by ec2-user and
# mode 0600 — nobody else on the host (or in the group) can read the
# GitHub App private key.
chown ec2-user:ec2-user ${PEM_REMOTE_FILE}
chmod 0600 ${PEM_REMOTE_FILE}
fi

cat <<'ENVFILE' | base64 -d > ${APP_ENV_REMOTE_FILE}
${APP_ENV_B64}
ENVFILE
# Env file also holds secrets (LLM API key, OAuth client secret, web
# session secret). Lock it down to the service user.
chown ec2-user:ec2-user ${APP_ENV_REMOTE_FILE}
chmod 0600 ${APP_ENV_REMOTE_FILE}

cat >/etc/systemd/system/${SERVICE_NAME}.service <<'UNIT'
[Unit]
Description=reviewbot interactive reviewboard web app
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=ec2-user
Group=ec2-user
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_ENV_REMOTE_FILE}
Environment=PORT=8080
Environment=PYTHONUNBUFFERED=1
ExecStart=${APP_DIR}/.venv/bin/reviewbot-web
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now ${SERVICE_NAME}.service
EOF
)"

# ---- launch instance --------------------------------------------------------
echo "==> launching $INSTANCE_TYPE instance (private subnet, no public IP)"
INSTANCE_ID="$(aws ec2 run-instances \
  --region "$AWS_REGION" \
  --image-id "$AMI_ID" \
  --instance-type "$INSTANCE_TYPE" \
  --key-name "$KEY_NAME" \
  --subnet-id "$SUBNET_ID" \
  --security-group-ids "$SG_ID" \
  --no-associate-public-ip-address \
  --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLUME_SIZE_GB},VolumeType=gp3}" \
  --user-data "$USER_DATA" \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${STACK_NAME}},{Key=stack,Value=${STACK_NAME}}]" \
  --query 'Instances[0].InstanceId' --output text)"
echo "    $INSTANCE_ID"

echo "==> waiting for instance to reach running state"
aws ec2 wait instance-running --region "$AWS_REGION" --instance-ids "$INSTANCE_ID"

PRIVATE_IP="$(aws ec2 describe-instances \
  --region "$AWS_REGION" \
  --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)"

# ---- write state ------------------------------------------------------------
jq -n \
  --arg region        "$AWS_REGION" \
  --arg stack         "$STACK_NAME" \
  --arg instance      "$INSTANCE_ID" \
  --arg subnet        "$SUBNET_ID" \
  --arg vpc           "$VPC_ID" \
  --arg sg            "$SG_ID" \
  --arg key           "$KEY_NAME" \
  --arg key_file      "$KEY_FILE" \
  --arg private_ip    "$PRIVATE_IP" \
  --arg service_name  "$SERVICE_NAME" \
  --arg app_env_local "$APP_ENV_LOCAL_FILE" \
  --arg pem_remote    "$PEM_REMOTE_FILE" \
  --argjson sg_ours   "$SG_CREATED_BY_US" \
  --argjson key_ours  "$KEY_CREATED_BY_US" \
  '{region:$region, stack:$stack, instance_id:$instance, subnet_id:$subnet, vpc_id:$vpc, sg_id:$sg, key_name:$key, key_file:$key_file, private_ip:$private_ip, service_name:$service_name, app_env_local:$app_env_local, pem_remote:$pem_remote, sg_created_by_us:$sg_ours, key_created_by_us:$key_ours}' \
  > "$STATE_FILE"

cat <<EOF

==> done.
    instance:    $INSTANCE_ID
    private ip:  $PRIVATE_IP   (reachable via private network/VPN)
    ssh:         ssh -i "$KEY_FILE" ec2-user@$PRIVATE_IP
    app:         http://$PRIVATE_IP:8080

cloud-init installs Python, clones the repo, writes ${APP_ENV_REMOTE_FILE},
and enables ${SERVICE_NAME}.service at boot. It typically takes 2-4 minutes
after the instance hits 'running'. Watch progress with:
    ssh -i "$KEY_FILE" ec2-user@$PRIVATE_IP 'sudo tail -f /var/log/cloud-init-output.log'

Once cloud-init is done, inspect the service with:
    ssh -i "$KEY_FILE" ec2-user@$PRIVATE_IP 'sudo systemctl status ${SERVICE_NAME}'
    ssh -i "$KEY_FILE" ec2-user@$PRIVATE_IP 'sudo journalctl -u ${SERVICE_NAME} -f'

state written to $STATE_FILE — do not delete it if you want destroy.sh to work.
EOF
