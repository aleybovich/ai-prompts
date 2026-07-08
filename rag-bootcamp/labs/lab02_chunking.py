"""
Lab 2 — Chunking strategies, written from scratch.

Implements three splitters and compares them on a real document:
  - fixed_size_chunks : every N characters, with overlap
  - recursive_chunks  : split on paragraphs -> lines -> sentences -> words (the default)
  - sentence_chunks   : pack whole sentences up to a size budget

Then shows why chunk-level retrieval beats whole-document retrieval.

Run it:   python lab02_chunking.py

We measure size in CHARACTERS to avoid a tokenizer dependency. Rule of thumb:
1 token ~= 4 characters of English, so ~1800 chars ~= ~450 tokens.
"""

import os
import re

import ragkit

# ~450 tokens and ~15% overlap, expressed in characters.
CHUNK_CHARS = 1800
OVERLAP_CHARS = 270


def fixed_size_chunks(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """Slide a fixed window over the raw text. Simple; may cut mid-word."""
    chunks, start = [], 0
    step = max(1, size - overlap)
    while start < len(text):
        end = start + size
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):        # already reached the end; don't emit a sliver
            break
        start += step
    return chunks


def recursive_chunks(text, size=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    """
    Split on the biggest natural boundary that keeps pieces under `size`,
    falling back to finer boundaries: paragraphs -> lines -> sentences -> words.
    This is what LangChain's RecursiveCharacterTextSplitter does.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(t, seps):
        if len(t) <= size or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        out, buf, buflen = [], [], 0
        for part in t.split(sep):
            add = len(part) + (len(sep) if buf else 0)
            if buf and buflen + add > size:
                out.append(sep.join(buf))       # rejoin parts, no trailing sep
                buf, buflen = [], 0
            if len(part) > size:                # a single part too big: go finer
                if buf:
                    out.append(sep.join(buf))
                    buf, buflen = [], 0
                out.extend(split(part, rest))
            else:
                buf.append(part)
                buflen += add
        if buf:
            out.append(sep.join(buf))
        return out

    raw = [c.strip() for c in split(text, separators) if c.strip()]
    # add overlap by prepending the tail of the previous chunk
    chunks = []
    for i, c in enumerate(raw):
        if i > 0 and overlap:
            chunks.append(raw[i - 1][-overlap:] + " " + c)
        else:
            chunks.append(c)
    return chunks


def sentence_chunks(text, size=CHUNK_CHARS):
    """Greedily pack whole sentences until adding one would exceed `size`."""
    sentences = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) + 1 <= size:
            buf = (buf + " " + s).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return chunks


def preview(chunk, n=70):
    return re.sub(r"\s+", " ", chunk)[:n]


def main():
    docs = {d["id"]: d for d in ragkit.load_corpus()}
    # Concatenate a few docs into one longer "document" so chunking is visible.
    big = "\n\n".join(docs[i]["text"] for i in ["security.md", "api.md", "troubleshooting.md"])
    print(f"Source document: {len(big)} characters (~{len(big)//4} tokens)\n")

    for name, fn in [
        ("fixed_size", fixed_size_chunks),
        ("recursive", recursive_chunks),
        ("sentence", sentence_chunks),
    ]:
        chunks = fn(big)
        sizes = [len(c) for c in chunks]
        print(f"{name:11s}: {len(chunks):2d} chunks | sizes {min(sizes)}-{max(sizes)} chars")
        print(f"             chunk 0 starts: {preview(chunks[0])!r}")
        print(f"             chunk 1 starts: {preview(chunks[1], 60)!r}\n")

    # --- Overlap in action (tiny, controlled example) ---------------------
    print("-" * 62)
    print("Overlap in action — a small example so you can see it (size=50, overlap=18)\n")
    demo = "A refund is issued within 14 days. It is returned to your original card."
    for label, ov in [("NO overlap ", 0), ("overlap=18 ", 18)]:
        print(f"  {label}:")
        for i, c in enumerate(fixed_size_chunks(demo, size=50, overlap=ov)):
            print(f"    chunk {i}: {c!r}")
    print("  -> With overlap, the tail of chunk 0 is repeated at the start of chunk 1,")
    print("     so a sentence split across the cut still appears whole in one chunk.")

    # --- Chunk-level vs whole-document retrieval --------------------------
    print("\n" + "-" * 62)
    print("Chunk-level retrieval is more precise than whole-document retrieval\n")
    embedder = ragkit.get_embedder()
    all_docs = ragkit.load_corpus()

    # whole documents
    doc_vecs = embedder.encode([d["text"] for d in all_docs])

    # SMALL chunks so a long doc actually subdivides and a sub-section can win
    chunk_texts, chunk_meta = [], []
    for d in all_docs:
        for c in recursive_chunks(d["text"], size=500, overlap=75):
            chunk_texts.append(c)
            chunk_meta.append(d["id"])
    chunk_vecs = embedder.encode(chunk_texts)

    query = "What should I do when the API returns 429?"
    qv = embedder.encode([query])[0]
    print(f"  Query: {query!r}\n")

    di, ds = ragkit.top_k(qv, doc_vecs, k=1)[0]
    print("  Best WHOLE DOCUMENT:")
    print(f"    ({ds:+.2f}) {all_docs[di]['id']} — {len(all_docs[di]['text'])} chars of text\n")

    ci, cs = ragkit.top_k(qv, chunk_vecs, k=1)[0]
    print("  Best CHUNK:")
    print(f"    ({cs:+.2f}) from {chunk_meta[ci]} — {len(chunk_texts[ci])} chars:")
    print(f"    {preview(chunk_texts[ci], 120)!r}")
    print("\n  The chunk hands the LLM a short, on-target passage instead of a")
    print("  whole document — cheaper to send and easier to answer from.")

    # --- Loading a REAL PDF with pypdf, then chunking it -------------------
    print("\n" + "-" * 62)
    print("Loading text out of a real PDF (pypdf), then chunking it\n")
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample.pdf")
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        pdf_text = "\n".join(page.extract_text() for page in reader.pages)
        print(f"  Extracted {len(pdf_text)} chars from {len(reader.pages)} page(s) of sample.pdf")
        print(f"  preview: {preview(pdf_text, 90)!r}")
        pdf_chunks = recursive_chunks(pdf_text, size=400, overlap=60)
        print(f"  -> split into {len(pdf_chunks)} chunk(s); chunk 0: {preview(pdf_chunks[0], 70)!r}")
        print("  Real documents (PDF, docx, html) become text FIRST, then chunk the")
        print("  same way. Always eyeball the extracted text before trusting it —")
        print("  bad parsing poisons everything downstream.")
    except ImportError:
        print("  (install pypdf — it is in requirements.txt — to run this part)")


if __name__ == "__main__":
    main()
