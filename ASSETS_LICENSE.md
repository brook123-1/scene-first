# Asset licensing

The repository-level [Apache-2.0 license](LICENSE) applies to project source code and documentation. It does **not** automatically license photographs, model weights, icons, avatar art, or other visual assets.

## Third-party model weights

| Paths | Origin | License | Redistribution status |
| --- | --- | --- | --- |
| `model_assets/face_detection_yunet_2023mar.onnx` | OpenCV Zoo YuNet | MIT | Verified for redistribution; retain `LICENSES/YUNET-MIT.txt`. |
| `model_assets/person_detection_mediapipe_2023mar.onnx` | OpenCV Zoo MP-PersonDet | Apache-2.0 | Verified for redistribution; retain `LICENSES/PERSON-DETECTION-APACHE-2.0.txt`. |

Checksums for the reviewed files are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The optional Pose Landmarker task bundle is downloaded at runtime to `.local/`; it is not distributed by this repository. Do not commit a downloaded task bundle unless its exact redistribution terms have first been verified.

## Deterministic project-source assets

The clean public repository contains these non-photographic project-source assets under Apache-2.0:

| Paths | Source method | Status |
| --- | --- | --- |
| `static/assets/safe-covers/*.png` | Pillow primitives from `scripts/generate_public_placeholders.py` | Deterministic geometric placeholders; no font, model, external artwork, brand element, or identifiable person. |
| `static/assets/app-icon.svg`, `app-icon-192.png`, `app-icon-512.png` | SVG/Pillow primitives from the same generator | Deterministic v0.1.0 placeholder icon. |
| `assets/avatar_families/generic/**` | Restricted SVG geometry plus `scripts/build_pose_avatar_assets.py` | Synthetic registry/debug fixture; not a designed identity-bearing avatar pack. |

The PNG files contain no EXIF/GPS metadata. CI regenerates the placeholders in a temporary directory and requires byte-for-byte PNG equality plus newline-normalized SVG text equality with the committed outputs.

## Assets outside the public allowlist

The public repository is assembled from the exact paths in `scripts/public-export-files.txt`. A visual pack or other asset present only in a private development source is not part of this distribution and receives no license from this repository.

Avatar packs are pluggable assets and may use licenses independent of the core source code. A user or redistributor must verify the license, authorship/source terms, commercial-use conditions, and attribution requirements for every imported pack.

## User content

Photos uploaded by users, selected crops, generated outputs, previews, annotations, and local benchmark results remain the user's content. They are never licensed by this repository. Do not submit them to issues or pull requests without the rights and informed consent needed for permanent public distribution.
