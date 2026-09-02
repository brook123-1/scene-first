# Privacy model

Scene First is source code for a local or self-hosted tool. The project itself does not operate an account system, analytics service, telemetry endpoint, or shared photo database. A person who deploys it remotely becomes responsible for that deployment's access control, notices, retention, logs, subprocessors, and legal obligations.

## What the browser sends

### Detection copy

The browser creates a resized, metadata-free detection copy and sends it to FastAPI. When FastAPI runs at `127.0.0.1`, this remains on the same computer. When FastAPI is remote, the copy leaves the device and reaches that server. The detection route decodes the copy, runs face/person detection, logs dimensions/timing/counts, and does not save the copy as an upload.

### Traditional Master

The compatibility path uploads the full selected/cropped original to FastAPI. The server normalizes orientation, creates metadata-free PNG/JPEG files, and stores working images, previews, jobs, and outputs in `.local/app/`. If an external provider is selected, the normal public UI sends a padded selected-person crop to that provider and composites the result on the server. The backend's `cloud_scope=full` compatibility option can send the full working image to an external provider; do not enable it without an explicit privacy decision.

### Local Master

When `SCENE_FIRST_LOCAL_MASTER` is enabled and the browser supports the required APIs, the full-resolution source stays in browser memory. Confirmed per-person crops are uploaded to FastAPI, validated so they are not the full image, then sent to the selected external provider. The browser downloads the returned patches and performs final full-resolution compositing locally. A resized detection copy still reaches FastAPI.

### Fully local provider

`LocalIllustrationProvider` and safe-cover mode do not call an external AI provider. When the whole application runs on the same computer, image processing remains on that computer. A remote self-hosted FastAPI instance is not “device local”: uploaded content reaches that server.

## External providers

BYOK adapters exist for OpenAI, Gemini, fal.ai, Volcengine Ark, Black Forest Labs, and DashScope/Qwen. Depending on route and adapter, a selected crop, mask, prompt, request metadata, and returned image URL/data may be sent or received. Provider terms, geographic routing, retention, abuse monitoring, training policy, safety processing, and price are controlled by that provider—not this project.

API keys are loaded server-side from environment variables or `.env.local`. They are not intentionally returned to the browser. The local Settings endpoint can write a key to `.env.local`; it is disabled by public-mode route guards and must never be exposed without authentication and network controls.

## Metadata and output

Input orientation is applied with EXIF transpose. Saved working images, previews, and exports are encoded without carrying the original EXIF block. This removes embedded image metadata from those outputs, but it does not remove identifying visual content in the pixels or information added later by another application.

## Local storage and cleanup

Runtime data lives under `.local/app/`, including uploads, masters, previews, output images, job JSON, per-person jobs, local benchmark data, and a cost ledger. Git and Docker build contexts exclude `.local`. The current prototype does not provide a complete retention scheduler or secure-erasure guarantee. Stop the app and delete the relevant `.local/app/` data yourself when it is no longer needed; back up only with the subject's permission.

Server logs include operational information such as generated IDs, dimensions, timing, detection counts, errors, provider/model labels, and cost estimates. They are not intended to include image bytes or API keys, but error responses from third parties can contain sensitive details; sanitize logs before sharing.

## Limits

Detection is fallible. Always inspect the full image, including reflections, screens, posters, badges, text, and bystanders. Replacing a face does not hide body shape, clothing, location, companions, or other identifying context.

This project helps reduce unwanted identity exposure; it does not guarantee anonymity.
