---
description: Deploy the current working tree to the serge EC2 box (in-place update + restart)
allowed-tools: Bash(./aws/update.sh), Bash(aws sso login:*), Bash(aws sts get-caller-identity:*)
---

Deploy the local working tree to the running `serge` / `reviewbot-web` EC2
instance by running the in-place updater.

Deployment target (recorded in `aws/.deploy-state.json`):
- instance: `i-0569c3492dcf2e4a4`, region `eu-north-1`
- AWS profile: `HF-Sandbox-access-754289655784`

`aws/update.sh` rsyncs the working tree to the box, reinstalls deps, rewrites
the env file, and restarts the service. It reads the profile from the state
file and will tell you to run `aws sso login` if credentials are expired.

Steps:

1. Run the deploy:

   ```bash
   ./aws/update.sh
   ```

2. If it fails with an AWS credentials / SSO error, run:

   ```bash
   aws sso login --profile HF-Sandbox-access-754289655784
   ```

   then re-run `./aws/update.sh`.

3. Report the final `systemctl status` output and the app URL the script
   prints. Do not commit or push anything — this command only deploys.
