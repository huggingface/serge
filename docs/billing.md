# HF inference costs

The task and review pages display the latest HF-reported inference cost when
the configured HF key can read billing for the account being charged. This
includes web reviews, webhook reviews, and integration-failure tasks. Other
providers and HF keys without billing access keep working without a cost display.

Each new job sends one `X-HF-Session-id` across all of its HF Router calls:
classification, agent turns, retries, normalization repairs, and follow-up
verification rounds share the same ID. The ID travels to isolated runners in
their configuration. It does not change prompt-cache settings.

Serge reads `billing/usage-by-inference-session` for the configured billing
organization (`LLM_BILL_TO`), or the token owner's account when there is no
organization. It sums that session's `costCents` across monthly periods and
converts cents to USD. These are provider-reported costs, not estimates from
token counts. Kubernetes/GPU verification infrastructure costs are not included.

The pages refresh once a minute, including after a job finishes, because HF
reconciles costs asynchronously. `pending` means HF has not returned the
session yet; it does not mean the job was free. `reported` is the latest HF
amount, not a guarantee that billing is finalized. If a refresh fails, a saved
amount remains visible as `stale`. The last amount and fetch time survive a
Serge restart. Billing is refreshed on page/API reads and cached for 60 seconds;
there is no background invoice collector.

`GET /tasks/{owner}/{repo}/{job_id}/info`, its OIDC-authorized `/status` sibling,
and `GET /reviews/{owner}/{repo}/{number}/{job_id}/info` include a `billing`
object with `status`, `cost_usd`, and (when reported) `request_count` and
`reported_at`. Authorization is checked before fetching billing. Responses
contain only that job's cost, never the organization's other sessions or keys.

For organization billing, grant the configured token `org.billing.read` for
the billed organization. Reads use the job's original provider configuration,
with its current key, so rotation is supported. No new copies of the key are
stored with billing data.

Deploy both the web image and runner images containing this change. Jobs from
before that deployment were not tagged and cannot be backfilled from this API.

References:

- [HF billing API specification](https://huggingface.co/.well-known/openapi.json)
- [HF provider cost reconciliation](https://huggingface.co/docs/inference-providers/register-as-a-provider#4-billing)
- [HF session-header implementation](https://huggingface.co/spaces/smolagents/ml-intern/blob/1ae7e9ba9999badd74cd624adc7bb5af58446bd9/agent/core/prompt_caching.py)
