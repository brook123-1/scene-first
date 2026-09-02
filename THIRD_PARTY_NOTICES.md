# Third-party notices

This inventory covers direct runtime/development dependencies and bundled binary assets as reviewed on 2026-09-01. Transitive packages remain governed by their own license files in installed distributions and lockfiles.

## Bundled model weights

### OpenCV Zoo YuNet

- File: `model_assets/face_detection_yunet_2023mar.onnx`
- SHA-256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- Source: <https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet>
- License: MIT; copyright 2020 Shiqi Yu.
- Status: commercial use and redistribution are permitted with the copyright and license notice retained. See `LICENSES/YUNET-MIT.txt`.

### OpenCV Zoo MP-PersonDet

- File: `model_assets/person_detection_mediapipe_2023mar.onnx`
- SHA-256: `47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f`
- Source: <https://github.com/opencv/opencv_zoo/tree/main/models/person_detection_mediapipe>
- License: Apache-2.0.
- Status: commercial use and redistribution are permitted subject to Apache-2.0. See `LICENSES/PERSON-DETECTION-APACHE-2.0.txt`.

### Optional pose task bundle

`app/pose_avatar/pose_adapters.py` can download a Google MediaPipe Pose Landmarker task bundle into `.local/`. The bundle is not committed or packaged in the container. The repository does not make a redistribution claim for that file; users obtain it from the upstream URL for local use.

## Python direct dependencies

| Package | Version pinned | Publisher source | Declared license |
| --- | --- | --- | --- |
| FastAPI | 0.139.2 | <https://pypi.org/project/fastapi/> | MIT |
| CairoSVG | 2.8.2 | <https://github.com/Kozea/CairoSVG> | LGPL-3.0-or-later |
| HTTPX | 0.28.1 | <https://pypi.org/project/httpx/> | BSD-3-Clause |
| NumPy | 2.2.6 | <https://pypi.org/project/numpy/> | BSD-3-Clause (binary wheels may include separately licensed libraries) |
| opencv-python-headless | 4.12.0.88 | <https://pypi.org/project/opencv-python-headless/> | Apache-2.0; wheel notices apply |
| Pillow | 12.3.0 | <https://pypi.org/project/pillow/> | MIT-CMU |
| pillow-heif | 1.4.0 | <https://github.com/bigcat88/pillow_heif/tree/v1.4.0> | BSD-3-Clause package; its bundled codec libraries have their own notices and copyleft terms |
| python-dotenv | 1.2.2 | <https://pypi.org/project/python-dotenv/> | BSD-3-Clause |
| python-multipart | 0.0.32 | <https://pypi.org/project/python-multipart/> | Apache-2.0 |
| pytest | 9.1.1 | <https://pypi.org/project/pytest/> | MIT |
| Uvicorn | 0.51.0 | <https://pypi.org/project/uvicorn/> | BSD-3-Clause |

Installed wheels and container images are aggregations. Redistributors of built binaries must retain the license material shipped by those packages; this notice is not a substitute for it.

## npm and GitHub Actions

Direct npm packages are [`@cloudflare/containers`](https://www.npmjs.com/package/@cloudflare/containers) (MIT OR Apache-2.0), [`@cloudflare/workers-types`](https://www.npmjs.com/package/@cloudflare/workers-types) (MIT OR Apache-2.0), [Playwright](https://github.com/microsoft/playwright) (Apache-2.0), [TypeScript](https://github.com/microsoft/TypeScript) (Apache-2.0), and [Wrangler](https://github.com/cloudflare/workers-sdk) (MIT OR Apache-2.0). The lockfile records all transitive versions and declared SPDX expressions, including LGPL components used by Sharp/libvips.

CI uses official `actions/checkout`, `actions/setup-python`, and `actions/setup-node` actions, each under MIT, plus the Gitleaks action/tool under MIT. Dependabot monitors action updates.

## Container base

The Dockerfile uses the [official Python image](https://hub.docker.com/_/python), tag `python:3.12-slim`. A deployed image also contains Debian packages and Python wheels under their respective licenses. Review the generated image's package notices when distributing a prebuilt image.

## Visual assets

See [ASSETS_LICENSE.md](ASSETS_LICENSE.md). The geometric safe covers, PWA icon, and `generic` avatar fixture in the clean public repository are deterministic project-source assets, not third-party notices. User-imported avatar packs keep their own independent licenses.
