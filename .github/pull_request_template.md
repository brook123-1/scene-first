## Summary

Describe the user-visible change and why it is scoped.

## Privacy and data flow

State whether image bytes, crops, masks, metadata, credentials, storage, logs, or external providers change.

## Verification

- [ ] Relevant Python tests pass.
- [ ] TypeScript typecheck passes when Cloudflare code changed.
- [ ] Avatar schema validation passes when avatar code/assets changed.
- [ ] I used synthetic fixtures; no private photo, crop, personal data, or `.local` screenshot is included.
- [ ] No key, token, cookie, authorization header, `.env` file, private domain, or deployment state is included.
- [ ] Every new dependency, model, font, icon, image, or copied snippet has first-party provenance and a redistribution-compatible license recorded.
- [ ] I reviewed whether the human-confirmation step, crop boundary, fallback, outside-mask verification, retention, or production behavior changed.
- [ ] Documentation and `.env.example` are updated where behavior/configuration changed.
- [ ] The change does not require a paid API or real photo in CI.

## Production behavior

Explicitly state `unchanged` or describe the intended change and rollback.
