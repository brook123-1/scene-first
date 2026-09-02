# Detection models bundled into the container

- `face_detection_yunet_2023mar.onnx`: OpenCV YuNet face detector.
- `person_detection_mediapipe_2023mar.onnx`: OpenCV Zoo MediaPipe person detector.

They are copied into the image at build time so a newly started Cloudflare
Container never needs to download model weights during a user's first scan.
The URLs and fallback download checks remain in `app/config.py` and
`app/detector.py` for local recovery.

Redistribution was verified from the model-specific OpenCV Zoo license files:

- YuNet: MIT, SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- MP-PersonDet: Apache-2.0, SHA-256 `47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f`.

See `THIRD_PARTY_NOTICES.md` and `LICENSES/` before replacing either file.
