# Configuration

FastAPI loads non-empty values from `.env.local`, then overlays non-empty process environment variables. Environment variables therefore win. Boolean truthy values are `1`, `true`, `yes`, and `on` (case-insensitive); anything else is false.

## Runtime and feature flags

| Name | Default | Allowed values | Meaning and privacy implication |
| --- | --- | --- | --- |
| `SCENE_FIRST_PUBLIC_MODE` | false | boolean | Hides local lab/settings/review/media routes. It reduces exposed routes but is not full authentication. Cloudflare Worker forces it to true. |
| `SCENE_FIRST_LOCAL_MASTER` | false locally | boolean | Enables crop-only Local Master endpoints and browser flow. Full original can stay in the browser; detection copy and external-provider crops still leave it. Checked-in Cloudflare configs set it to `1`. |
| `SCENE_FIRST_POSE_AVATAR_OVERLAY` | false | boolean | Enables pose enrichment and allows request-level experimental overlay opt-in. It downloads an optional pose task into `.local` when used. |

## Image limits and concurrency

| Name | Default | Allowed values | Meaning |
| --- | --- | --- | --- |
| `MAX_IMAGE_PIXELS` | `16000000` | positive integer | Maximum decoded full working image pixels. |
| `MAX_DETECTION_PIXELS` | `3200000` | positive integer | Maximum decoded detection-copy pixels. |
| `DETECTION_TILE_MIN_EDGE` | `1450` | positive integer | Minimum edge threshold used for tiled detection. |
| `DETECTION_ENABLE_TILES` | true | boolean | Enables tiled detection; disabling can reduce compute and reduce small-person recall. |
| `MAX_PARALLEL_PERSON_EDITS` | `2` | positive integer | Maximum provider edits in parallel per photo; can affect rate limits and cost bursts. |

Upload byte limits (30 MB full file, 8 MB Local Master person crop) and Local Master geometry limits are currently code constants rather than environment variables.

## Provider credentials and models

| Name | Default | Meaning |
| --- | --- | --- |
| `OPENAI_API_KEY` | empty | Enables OpenAI image edits. |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2` | Reviewed OpenAI model identifier. |
| `GEMINI_API_KEY` | empty | Enables Gemini image edits. |
| `GEMINI_IMAGE_MODEL` | `gemini-3.1-flash-image` | Reviewed Gemini model identifier. |
| `FAL_KEY` | empty | Enables fal.ai. |
| `FAL_IMAGE_MODEL` | `fal-ai/nano-banana-pro/edit` | Must be one of the adapter-reviewed edit endpoints in `app/config.py`. |
| `ARK_API_KEY` | empty | Enables Volcengine Ark. |
| `ARK_IMAGE_BASE_URL` | `https://ark.cn-beijing.volces.com/api/plan/v3/images/generations` | Ark image endpoint. Changing it sends credentials/image data to a different host; treat as a security-sensitive override. |
| `ARK_IMAGE_MODEL` | `doubao-seedream-5.0-lite` | Ark model identifier. |
| `BFL_API_KEY` | empty | Enables Black Forest Labs. |
| `BFL_IMAGE_MODEL` | `flux-2-pro` | BFL endpoint/model identifier; code constrains it to `flux-*` or falls back. |
| `DASHSCOPE_API_KEY` | empty | Enables DashScope/Qwen. |
| `QWEN_IMAGE_MODEL` | `qwen-image-2.0` | Qwen model identifier. |

Never set credentials to `PLACEHOLDER`: leave them empty until valid, keep `.env.local` ignored, and use platform secret stores for remote deployments.

## Optional local cost ledger

| Name | Default | Allowed values | Meaning |
| --- | --- | --- | --- |
| `*_ESTIMATED_CNY` | `0` | non-negative number | Per-generation reference for the named provider. It is manually supplied, not live pricing. |
| `ARK_BILLING_MODE` | `agent_plan` | `agent_plan` or `payg` | Chooses how Ark usage is labelled in the local ledger. |
| `COST_LEDGER_USD_CNY` | `7.2` | positive number | Manual reference conversion for local bookkeeping, not financial data. |

The six estimate variables are `OPENAI_ESTIMATED_CNY`, `GEMINI_ESTIMATED_CNY`, `FAL_ESTIMATED_CNY`, `ARK_ESTIMATED_CNY`, `BFL_ESTIMATED_CNY`, and `QWEN_ESTIMATED_CNY`.

## Deployment and manual verification

| Name | Default | Scope |
| --- | --- | --- |
| `SCENE_FIRST_TEST_PASSWORD` | empty | Optional Worker/legacy tunnel access control. Store as a secret. |
| `SCENE_FIRST_ADMIN_RESTART_TOKEN` | empty | Optional Bearer token for a narrow Container restart endpoint. Store temporarily as a secret and remove after use. |
| `SCENE_FIRST_STAGING_URL` | empty | Manual staging browser test target; not read by FastAPI. |
| `SCENE_FIRST_STAGING_ACCESS_CODE` | empty | Manual staging test credential. |
| `SCENE_FIRST_TEST_PROXY` | empty | Optional Playwright proxy URL for manual tests. |
| `SCENE_FIRST_TEST_PROXY_USERNAME` | empty | Optional manual-test proxy username. |
| `SCENE_FIRST_TEST_PROXY_PASSWORD` | empty | Optional manual-test proxy password. |

The checked-in Wrangler files deliberately omit account IDs, routes, domains, and secrets. Configure them in your own Cloudflare account and review [PRIVACY.md](../PRIVACY.md) before remote deployment.
