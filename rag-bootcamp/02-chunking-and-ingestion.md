# Lesson 2 — Chunking and ingestion

**Goal:** learn to split documents into good retrieval units ("chunks"), why
chunk size and overlap matter, and how to load real files (including PDFs).

**Ingestion** is the offline pipeline that prepares your corpus: **load →
chunk → embed → store**. This lesson covers load and chunk; Lesson 3 covers
store. Chunking is where beginners most often leave retrieval quality on the
table, so it is worth doing well.

---

## 1. Why not embed whole documents?

Two reasons you cannot just embed each document as one vector:

1. **Context limits.** Embedding models cap their input (Lesson 1). A 30-page PDF
   will be silently truncated — most of it never gets embedded at all.
2. **Precision.** One vector for a whole document blurs everything together. A
   question about the refund policy shouldn't have to compete with the entire
   employee handbook. Smaller units retrieve more precisely, and you feed the
   LLM a tight, relevant passage instead of 30 pages.

So we split documents into **chunks** — passages of roughly a paragraph to a
page — embed each chunk, and retrieve chunks.

## 2. The two dials: chunk size and overlap

**Chunk size.** Too small and a chunk loses the context that makes it meaningful
("It costs $8/month" — *what* costs $8?). Too large and it dilutes: one chunk
covers many topics, so its embedding is a muddy average and retrieval gets vague.

A widely used starting point: **~400–512 tokens per chunk** (roughly 1,600–2,000
characters, since a token is ~4 characters of English). Treat this as a dial to
tune per corpus, not a law. Dense reference docs often want smaller chunks;
flowing narrative wants larger.

**Overlap.** If you cut cleanly at boundaries, an idea that straddles two chunks
gets split — the definition lands in chunk A and the rule that uses it in chunk
B, and neither chunk alone answers the question. **Overlap** repeats the last
slice of one chunk at the start of the next (commonly **10–20%**), so a complete
thought survives somewhere.

```
Document:  [....... idea that spans the boundary .......]
No overlap: [ chunk A ][ chunk B ]        idea is split, both halves incomplete
Overlap:    [ chunk A ....]
                 [.... chunk B ]          the overlap keeps the idea whole in B
```

## 3. Chunking strategies, from simple to smart

| Strategy | How it splits | When to use |
| --- | --- | --- |
| **Fixed-size** | Every N characters/tokens, with overlap | Simplest; fine for uniform plain text. Can cut mid-sentence. |
| **Recursive character** | Tries paragraphs, then sentences, then words — whatever keeps chunks under the size limit | The **sensible default** for general text (LangChain's `RecursiveCharacterTextSplitter` popularized it). |
| **Structure-aware** | Splits on Markdown headings, HTML tags, or code structure | Best when documents have strong structure (docs, wikis, code). |
| **Semantic** | Groups sentences by embedding similarity, cutting at topic shifts | Can help on topically diverse documents, but benchmark results are mixed and it adds embedding cost at ingestion. |
| **Late chunking** | Embeds the *whole* document first (long-context model), then splits the token embeddings and pools each chunk | Preserves cross-chunk context (pronouns, references). Newer; needs a long-context model. |

Start with **recursive character** splitting. Move to structure-aware or semantic
only if evaluation (Lesson 7) shows retrieval is missing things.

## 4. Metadata: attach it at ingestion

Every chunk should carry **metadata** — at minimum its source document and
position, and often a date, author, category, or permission tag. You need it for:

- **Citations** — "this claim came from `pricing.md`."
- **Filtering** — "only search docs from this year / this team" (Lesson 3).
- **Debugging** — when an answer is wrong, you trace it to the exact chunk.

Store metadata alongside the vector. It is cheap at ingestion and painful to
add later.

## 5. Loading real files

Markdown and text are easy. The messy one is **PDF**. Options in 2026:

- **`pypdf`** — pure-Python, free, fine for clean digital PDFs. Used in the lab.
- **`PyMuPDF`** — faster text/image extraction, with a `pymupdf4llm` helper for
  Markdown output; lighter on layout/tables than Docling.
- **`Docling`** (IBM, free/open) — much better at **tables** and complex layout;
  the strong open choice for messy PDFs, scans, and financial/legal documents.
- **LlamaParse / Reducto** (paid) — hosted parsers for the hardest documents.

Rule of thumb: clean PDFs → `pypdf`; table-heavy or scanned PDFs → Docling. Bad
extraction poisons everything downstream, so it is worth checking the text a
parser produces before you trust it.

## 6. Lab 2 — chunk it three ways

[`labs/lab02_chunking.py`](./labs/lab02_chunking.py) implements fixed-size,
recursive-character, and sentence chunking from scratch (no framework), runs them
on a corpus document, and shows:

- how many chunks each produces and where the boundaries fall,
- the effect of overlap,
- how a precise query retrieves a *tight* chunk instead of a whole document.

```bash
cd labs && source .venv/bin/activate
python lab02_chunking.py
```

Writing the splitters by hand (they are ~15 lines each) demystifies what
frameworks do for you later.

## Key takeaways

- Chunk because of **context limits** and **retrieval precision**.
- Tune two dials: **size** (~400–512 tokens to start) and **overlap** (10–20%).
- **Recursive character** splitting is the default; structure-aware and semantic
  are upgrades to reach for when evaluation demands them.
- Attach **metadata** (source, position, date, tags) to every chunk at ingestion.
- Parse PDFs with `pypdf` (clean) or **Docling** (messy/tables); verify the
  extracted text.

## Further reading

- Late chunking — <https://jina.ai/news/late-chunking-in-long-context-embedding-models/>
- Pinecone chunking guide — <https://www.pinecone.io/learn/chunking-strategies/>

Next: **[Lesson 3 — Vector databases and indexing](./03-vector-stores-and-indexing.md)**.
