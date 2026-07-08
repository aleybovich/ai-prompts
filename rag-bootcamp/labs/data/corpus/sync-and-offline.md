# Sync and offline mode

Nimbus Notes keeps your notes in sync across every device on your account. Sync
runs automatically in the background whenever you are online. You never have to
press a "sync" button, though you can force a sync from the app menu.

## How sync works

Every note change is saved locally first, then uploaded to the Nimbus cloud and
pushed to your other devices. A small cloud icon in the corner of the app shows
sync status: a checkmark means everything is up to date, a spinning arrow means a
sync is in progress, and a warning triangle means sync is paused or failed.

## Working offline

Nimbus works fully offline. When you have no internet connection, you can still
read, create, edit, and delete notes. Your changes are stored on the device and
uploaded automatically the next time you come online. There is no separate
"offline mode" to turn on — it is always available.

## Conflicts

If the same note is edited on two devices while one of them is offline, Nimbus
may detect a conflict when they both sync. Rather than silently overwriting your
work, Nimbus keeps both versions: the current note plus a "Conflicted copy" note
placed in the same notebook. You can then merge them by hand and delete the copy.

## Sync troubleshooting

If the warning triangle appears and sync will not complete, first check that you
are online and signed in. If sync is still stuck, sign out and back in, which
rebuilds the local sync state. Note that signing out removes locally cached notes
that have already been uploaded, so make sure the sync completed before signing
out. If a single note refuses to sync, it is usually because an attachment
exceeds the 200 MB per-file limit.
