# Security policy

## Supported versions

Before the first public release, security fixes are made only on the repository's default development branch. After `v0.1.0`, the latest released minor line and the default branch are the intended supported versions; older experimental snapshots receive no guaranteed fixes.

## Report a vulnerability privately

Do not open a public issue for a vulnerability involving credentials, private photos, path traversal, arbitrary file access, authentication bypass, provider request leakage, or remote deployment controls.

The maintainer must enable GitHub **Private Vulnerability Reporting** before making the repository public. Once enabled, use the repository's **Security → Advisories → Report a vulnerability** flow. No security email address is currently published, so this repository does not invent one. Until Private Vulnerability Reporting is enabled, contact the maintainer privately through the GitHub profile and share only the minimum sanitized reproduction details.

Never attach an API key, cookie, `.env.local`, private photo, face/head crop, unredacted log, deployment token, or personal data to a report. Use synthetic fixtures and redact IDs, paths, domains, and headers.

## Secret handling

- Keep keys in `.env.local`, environment variables, or the deployment platform's secret store.
- Never place keys in `wrangler*.jsonc`, source code, screenshots, issue comments, test fixtures, or shell history.
- Use a separate low-privilege key for development where the provider supports it.
- If a key leaks, revoke/rotate it at the provider first, inspect access logs, remove it from the current tree, and assess Git history before publishing a replacement commit. Deleting one line in a new commit does not remove an old secret from history.

## Image and file-processing risk

Image decoders process attacker-controlled files. Keep Python, Pillow, pillow-heif, OpenCV, CairoSVG, Node.js, and container packages updated. The application enforces byte/pixel limits, normalizes images, and avoids trusting client crop coordinates without validation, but it has not undergone an independent security audit. Run it as an unprivileged user, bind locally unless remote access is intentional, isolate public deployments, and do not mount sensitive host directories.

## Dependencies and disclosure scope

Dependabot covers pip, npm, and GitHub Actions. CI compiles and tests without keys. Reports about an upstream provider or library should also follow that project's disclosure process; please tell this project privately when its configuration remains affected.
