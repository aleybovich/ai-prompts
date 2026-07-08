# Troubleshooting common problems

This page lists fixes for the problems people run into most often.

## I forgot my password

Use **Forgot password** on the sign-in screen to get a reset link by email. Note
that this resets your *account* password. It does **not** recover an end-to-end
encrypted notebook — those are protected by a separate passphrase that Nimbus
cannot reset. If you forget an E2EE passphrase, those notes cannot be recovered.

## Sync is stuck

If the sync icon shows a warning triangle, first confirm you are online and
signed in. Then try signing out and back in, which rebuilds local sync state.
A single note that will not sync is usually caused by an attachment over the
200 MB per-file limit — remove or shrink the attachment.

## Search is missing recent notes

Search uses a local index that occasionally needs rebuilding, for example after a
large import. Go to **Settings > Search > Rebuild index**. Rebuilding can take a
few minutes on large accounts; search results will be incomplete until it
finishes.

## The app is slow to start

A very large local cache can slow startup. In **Settings > Storage** you can
clear the local cache of already-synced notes; they re-download as you open them.
Make sure sync has completed before clearing the cache.

## Attachments will not upload

Attachments must be 200 MB or smaller per file. Free accounts also have a total
storage cap of 1 GB; if you hit it, uploads fail until you free space or upgrade
to Pro (50 GB).

## API calls fail with 429

`429 Too Many Requests` means you exceeded the API rate limit (120 requests per
minute on Pro, 600 on Team). Read the `Retry-After` header and wait that many
seconds before retrying.
