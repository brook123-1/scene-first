# Development guide

## Repository layout

| Path | Purpose |
| --- | --- |
| `app/` | FastAPI routes, detection, image operations, job stores, providers, cost ledger, Master/Local Master, pose overlay. |
| `static/` | Dependency-light browser UI, service worker, crop/detection/Local Master clients, icons and safe-cover assets. |
| `cloudflare/` | Worker and Container bridge for optional self-hosting. |
| `assets/avatar_families/` | Pose-aware avatar manifests, anchors, constraints, and artwork. |
| `model_assets/` | Reviewed OpenCV model weights copied into container images. |
| `scripts/` | Windows setup/start/test, public export, deterministic asset generation, and avatar validation. |
| `tests/` | Python unit/API tests plus local/manual browser scripts. |
| `docs/` | Public architecture, privacy, setup, provider, and release documentation. |
| `.local/app/` | Runtime state; private, generated, ignored, and never a fixture source. |

## Local setup

On Windows with Python 3.12 and Node.js 22+:

```powershell
npm install
npm run app:setup
npm run app:start
```

`app:setup` creates `.venv` and installs `requirements.txt`. The app binds only to `127.0.0.1:8765`. Environment variables override non-empty `.env.local` values.

For a non-standard Python installation, set `SCENE_FIRST_PYTHON` to a Python 3.12 executable before `npm run app:setup`. The setup script otherwise checks the Windows `py` launcher, `python`, and an optional Codex desktop runtime discovered from the current user profile.

## Backend and frontend

`app.main:app` is the FastAPI entry point. It serves the static UI and JSON/multipart endpoints. The browser code is plain JavaScript/CSS/HTML—there is no frontend bundle step. `static/detection-client.js` prepares the smaller detection copy; `static/local-master.js` implements crop-only full-resolution browser processing; `static/app.js` coordinates the product flow.

Lab-only pages such as review, preflight, settings, cost ledger, benchmark media, and pose playground are guarded when `SCENE_FIRST_PUBLIC_MODE` is enabled. Public mode is a route-reduction control, not a complete authentication or deployment hardening system.

## Image and job lifecycle

1. The browser applies orientation/crop and makes a metadata-free, pixel-limited detection copy.
2. `/api/detect` decodes it and uses the detector fusion pipeline. The response contains candidate regions and warnings, not a saved image URL.
3. The user selects, deselects, adds, or adjusts regions. `selection_confirmed=true` records that the candidate set was reviewed.
4. Traditional Master uploads the working original to `/api/images/{image_id}/master`; Local Master retains the full-resolution source in the browser and submits confirmed person crops to `/api/local-person-jobs`.
5. A provider returns a patch. Provider failures become a safe local fallback rather than silently leaving a selected person unchanged.
6. Masks constrain local compositing. `outside_mask_is_exact` checks that pixels outside the combined mask did not change.
7. Metadata-free outputs and job JSON live under `.local/app/`; the browser can export PNG/JPEG.

`JobStore` uses an in-process thread pool and JSON files. It is a prototype store, not a distributed queue, transactional database, multi-tenant isolation boundary, or retention service.

## Detection

`app/detector.py` combines YuNet face candidates with an OpenCV Zoo MediaPipe person detector, including tiling for large detection copies. Users must manually review results because small, profile, occluded, back-facing, and reflection cases can be missed. The two bundled ONNX files are copied into Docker; local recovery can download them from the configured OpenCV Hugging Face mirrors.

## Provider architecture

`app/providers.py` defines `ImageProvider.edit(crop, mask, ...)`. `LocalIllustrationProvider` is deterministic and keyless. External adapters read their key server-side and use explicit endpoints/request schemas. Returned images are resized/cropped and composited locally. See [PROVIDERS.md](PROVIDERS.md) before changing any payload.

## Master and Local Master

- **Traditional Master (default):** server receives and persists the complete working original, then performs provider calls and final compositing.
- **Local Master (flagged):** browser retains full resolution; FastAPI validates and processes only selected crops, while final high-resolution compositing happens in the browser. Detection still sends a resized copy to FastAPI. Unsupported browsers can ask the user before falling back to Traditional Master.

## Pose-aware Avatar Overlay

`app/pose_avatar/` contains manifest/schema validation, scene routing, 2D transforms, coverage masks, traces, and the local playground. The product route requires both `SCENE_FIRST_POSE_AVATAR_OVERLAY=true` and request-level `pose_aware_overlay=true`. The playground reads only local `.local/app/pose-avatar-p1` cases and does not change the production router. The public repository contains only the synthetic `generic` fixture; import a separately licensed pack for additional views. This remains experimental.

## Feature flags

All feature flags default off in local development:

- `SCENE_FIRST_PUBLIC_MODE`
- `SCENE_FIRST_LOCAL_MASTER`
- `SCENE_FIRST_POSE_AVATAR_OVERLAY`

See [CONFIGURATION.md](CONFIGURATION.md) for exact values and privacy impact.

## Testing

Safe automated checks:

```powershell
npm run app:test
npm run cf:typecheck
npm run app:validate-avatar -- assets/avatar_families/generic
npm run app:check-placeholders
```

`app:test` runs pytest. API tests use fakes/monkeypatches rather than paid providers. Browser scripts require local Chrome and, depending on the script, a running app. Staging verification additionally requires an explicitly configured URL and access code; it is not CI.

Never turn a private `.local` photo into a regression fixture. Reproduce geometry with synthetic images.

## Debug and runtime directories

Treat `.local/`, samples, benchmark sources, crops, outputs, and screenshots as private even when a file appears harmless. Do not serve debug media from an internet-facing deployment. The committed `generic` avatar images and `static/assets` placeholders are deterministic geometry rather than people; their generators/source shapes remain in the repository.

## Clean public export

The private development repository and clean public repository have separate histories. From a clean committed private-source ref, export only the exact paths in `scripts/public-export-files.txt`:

```powershell
npm run repo:export-public -- -Destination ..\scene-first-public
```

The destination must be empty and outside the private source repository. The script reads committed Git blobs, never copies `.git`, and fails if the output differs from its explicit allowlist. Audit the export before running `git init`; do not publish the private source repository or copy its history.
