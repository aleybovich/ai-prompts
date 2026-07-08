# Nimbus Notes changelog

Recent notable changes, newest first.

## Version 5.2 — March 2026

- Added optional **end-to-end encryption per notebook** for Pro and Team plans.
- Rebuilt the local search index for roughly 2x faster results on large accounts.
- The API moved to `/v2`; `/v1` is now deprecated.

## Version 5.1 — December 2025

- Real-time collaboration now shows each collaborator's cursor and selection.
- Added SCIM user provisioning for Team plans.
- Raised the per-file attachment limit from 100 MB to 200 MB.

## Version 5.0 — September 2025

- Introduced **team notebooks** owned by the organization.
- Added single sign-on (SSO) with SAML for Team plans.
- New JSON export format for developers.

## Version 4.4 — May 2025

- Full-text search inside PDFs and images (OCR) added to the Pro plan.
- Added scheduled weekly exports to cloud storage on the Team plan.

## Version 4.3 — February 2025

- Backlinks panel added: every note now shows which notes link to it.
- Public share links can now have an expiration date.

Older releases are listed in the archive. Nimbus follows semantic versioning:
the first number changes for major releases, the second for feature updates, and
patch releases (like 5.2.1) are shipped for bug fixes between these.
