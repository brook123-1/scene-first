# Releasing

Scene First uses [Semantic Versioning](https://semver.org/) after the first public release and a [Keep a Changelog](https://keepachangelog.com/) style changelog. The proposed first public version is `v0.1.0`; do not create it until every required launch checklist item is approved.

## Prepare a release

1. Start from a clean, protected default branch with CI green.
2. Review `CHANGELOG.md`; move relevant Unreleased entries into a dated version section.
3. Re-run the clean Windows Quick Start, Python tests, TypeScript typecheck, asset validator, current-tree secret scan, and complete-history secret scan.
4. Confirm no real photo, crop, private screenshot, `.local` file, key, private domain/config, or unclear model/asset entered the diff or history.
5. Review dependency/model licenses and refresh checksums if a binary changed.
6. Document configuration or data migrations and a rollback path.
7. Use only synthetic or rights-cleared screenshots. Strip metadata and inspect the final file before upload.
8. Draft release notes with status, privacy-impacting changes, known limitations, upgrade steps, and checksums for separately distributed binaries.

## Tag and GitHub Release

After explicit maintainer approval, create an annotated `vMAJOR.MINOR.PATCH` tag from the reviewed commit and create a GitHub Release from that tag. Do not attach `.env` files, runtime state, private images, browser traces, or locally built archives that have not been scanned.

For `v0.1.0`, label the project experimental and link `PRIVACY.md`, `SECURITY.md`, `ASSETS_LICENSE.md`, and the known limitations. A release is not a production deployment.

## Correct or withdraw a release

Git tags, cached archives, forks, and downloaded assets may survive after a GitHub Release is deleted. For a non-secret defect, mark the release as deprecated and publish a fixed patch. For a leaked key or personal image, revoke the key or contain the data first, preserve evidence privately, remove public artifacts, assess history/forks/caches, and follow the incident steps in `SECURITY.md`. Never assume deleting the Release makes exposed data private again.
