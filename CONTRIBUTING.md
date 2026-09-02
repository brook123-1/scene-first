# Contributing

Thank you for helping Scene First become safer and easier to run locally. Small, focused pull requests with synthetic tests are the easiest to review.

## First contribution

1. Fork the repository on GitHub.
2. Clone your fork and create a branch such as `fix/clear-detection-error`.
3. Install on Windows with `npm install` and `npm run app:setup`.
4. Make one scoped change. Preserve the human-confirmation step, privacy boundaries, and local fallback behavior.
5. Run:

   ```powershell
   npm run app:test
   npm run cf:typecheck
   npm run app:validate-avatar -- assets/avatar_families/generic
   ```

6. Commit with a short imperative message, push to your fork, and open a pull request against the default branch.

Explain the user-visible behavior, privacy impact, tests, and any new dependency or asset provenance. Update README/configuration/provider docs when behavior changes.

## Never submit

- API keys, passwords, cookies, tokens, authorization headers, certificates, `.env.local`, `.local/`, or deployment state;
- a real person's photo, face/head crop, generated derivative, private screenshot, EXIF/GPS data, or unredacted log;
- personal data or an absolute local path that identifies a user;
- proprietary model weights or any asset/dependency whose redistribution license is unclear;
- benchmark source photos or human annotations derived from private photos.

Use deterministic synthetic fixtures. If a bug only reproduces with private content, describe the geometry and create a synthetic substitute before submitting.

## Code and architecture principles

- A user must review automatic detections before a cloud edit.
- Default external calls remain crop-scoped; changing a provider payload requires an explicit privacy review.
- Never weaken outside-mask verification merely to accept a result.
- Keep runtime data in `.local/` and credentials outside Git.
- Mark experimental paths and keep feature flags off by default unless separately approved.
- Do not add provider pricing claims that will become stale.

## Pull-request review

Maintainers may ask for a smaller diff, privacy threat analysis, license evidence from a first-party source, or a synthetic regression test. Passing CI is necessary but not sufficient. By submitting a contribution, you agree that it is licensed under Apache-2.0 and that you have the right to submit it.
