# Providers and BYOK

External providers are optional. The keyless `local` provider is enough to exercise detection, review, masking, compositing, verification, and export.

## Contract

`ImageProvider.edit(crop, mask, *, subject_id, retry_nonce, prompt)` returns a Pillow image. Callers provide an immutable source crop and a local mask. The service layer catches provider failures, records the attempted provider/model, and uses a local safe fallback.

The provider abstraction does not make all privacy behavior equal. Some APIs receive a mask, others receive only the crop and natural-language instruction. Every external adapter transmits image pixels. Local mask compositing limits changes in the final file but cannot retract data already sent.

## Current adapters

| Provider name | Credential | Payload in the normal crop route | Mask sent? |
| --- | --- | --- | --- |
| `local` | None | No network request | Local only |
| `openai` | `OPENAI_API_KEY` | selected crop, prompt, request parameters | Yes |
| `gemini` | `GEMINI_API_KEY` | selected crop and prompt | No |
| `fal` | `FAL_KEY` | selected crop and prompt; mask only for reviewed mask-capable adapters | Model-dependent |
| `ark` | `ARK_API_KEY` | selected crop and prompt | No |
| `bfl` | `BFL_API_KEY` | selected crop and prompt | No |
| `qwen` | `DASHSCOPE_API_KEY` | selected crop and prompt | No |

Provider responses may contain hosted image URLs, which FastAPI downloads. DNS, network proxies, provider logging, safety review, content policy, output licensing, and billing follow the selected provider's current terms.

## Configure BYOK

Copy `.env.example` to `.env.local`, set one key, optionally override its reviewed model identifier, and restart FastAPI. Environment variables take precedence over `.env.local`. Do not put keys in client JavaScript or `wrangler*.jsonc`. For Cloudflare, use `wrangler secret put` or the dashboard secret UI.

The person who supplies the key obtains it from the provider, accepts that provider's terms, sets spend limits if available, and pays all charges. Scene First does not proxy free credits or promise a price.

## Adding a provider

1. Confirm the endpoint is an image-edit route and document its first-party API and license/terms source.
2. Define a fixed credential variable and reviewed model configuration; do not add arbitrary URL/key forwarding from the browser.
3. Send the smallest selected crop possible. Never change the public UI to full-image scope silently.
4. Set timeouts, bound response downloads, sanitize errors, and avoid logging headers or image data.
5. Add a fake-provider unit test that proves fallback and outside-mask behavior without a paid call.
6. Update `.env.example`, `docs/CONFIGURATION.md`, the privacy table, and third-party notices.
7. Manually validate against a synthetic image before asking maintainers to use consented private photos locally.

## Settings endpoint

The local Settings page can save a provider key/model to `.env.local`. It is a convenience for a trusted workstation, not a remote secret-management interface. `SCENE_FIRST_PUBLIC_MODE=true` hides lab/settings routes, but a real deployment still needs network access controls and platform secrets.
