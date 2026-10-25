# MIC baseline workflows and progress

The dashboard can dispatch `cloud-baselines.yml` in
`huggingface/transformers-mic` through Serge's existing `/dashboard/permission`
and `/dashboard/workflows/dispatch` API. The shared dashboard service token,
repository allowlist, actor write-permission check, workflow allowlist and
default-branch restriction all apply. No agent task or GPU pod is started by
Serge itself: the MIC workflow uses its existing ARC access.

The defaults now include the MIC repository and workflow. Deployments overriding
`DASHBOARD_REPOSITORIES` or `DASHBOARD_DISPATCH_WORKFLOWS` must add them while
preserving existing entries. Serge's App needs installation access and Actions
write permission on the MIC repository. The dashboard's read App needs Actions
and Contents read there too.

For live case progress, set `DASHBOARD_MIC_PROGRESS_URL` to the dashboard's
internal `http://trace-exporter.transformers-ci.svc.cluster.local:8000/mic-jobs/progress`.
The existing `DASHBOARD_API_TOKEN` authenticates this service-to-service relay.
Allow Serge's web pod to reach that Service through the cluster network policy.

`POST /dashboard/mic-progress` accepts a short-lived GitHub Actions OIDC token
with audience `serge-mic-baselines`. It verifies the signature, issuer, audience
and expiry with the existing OIDC verifier, then requires the MIC repository,
the `cloud-baselines.yml` workflow on `main`, and `workflow_dispatch`. The run ID
and attempt come from verified claims, not the request body. Progress payloads
have bounded counters, sequence numbers, stages and detail text. The dashboard
also binds the claimed run to the corresponding persisted job.

CLI bearer tokens and the ordinary dashboard service token cannot substitute
for a GitHub OIDC token at this public endpoint. Model execution never receives
the shared dashboard service token. Relay failures affect progress display only;
measurement validity and workflow completion are determined independently.

Model/baseline upload is implemented in MIC's dedicated trusted publisher job.
Serge does not hold or forward its Hub write token. Full setup and supported CLI
combinations live in `transformers-mic/docs/cloud-baselines.md`.
