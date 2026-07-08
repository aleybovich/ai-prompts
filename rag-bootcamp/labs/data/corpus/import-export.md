# Importing and exporting notes

Nimbus Notes is designed so your data is never locked in. You can bring notes in
from other apps and take them out whenever you like.

## Importing

From **Settings > Import**, Nimbus can import from:

- Markdown files and folders (each `.md` file becomes a note; folders become
  notebooks)
- Evernote `.enex` export files
- Notion workspace exports (the Markdown + CSV zip)
- Plain text and HTML files

During import, Nimbus preserves note titles, folder structure, attachments, and
creation dates where the source provides them. Very large imports (over 10,000
notes) run in the background and email you when they finish.

## Exporting

From **Settings > Export**, you can export a single notebook or your entire
account. Export formats are:

- **Markdown** — a zip of `.md` files, one per note, with attachments in a
  folder next to them. This is the recommended format for portability.
- **PDF** — a rendered PDF per note, good for archiving or printing.
- **HTML** — a self-contained website you can open in any browser.
- **JSON** — a structured export including note metadata, tags, and links, meant
  for developers who want to process notes programmatically.

Exports include your notes, attachments, tags, and links. Version history is not
included in exports. Exporting does not delete anything from Nimbus.

## Automating exports

Team plans can schedule an automatic weekly export to a connected cloud storage
account (such as Google Drive or Dropbox) as an extra backup.
