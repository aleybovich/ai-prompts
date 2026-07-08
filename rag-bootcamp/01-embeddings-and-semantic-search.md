# Lesson 1 — Embeddings and semantic search

**Goal:** understand how text becomes a vector of numbers, how "closeness" of
those vectors means "closeness of meaning," and build a working semantic search.

This is the heart of the *retrieval* half of RAG. Get comfortable here and the
rest of the course clicks into place.

---

## 1. From words to numbers

Computers cannot compare *meaning* directly. So we convert each piece of text
into a list of numbers — a **vector** — chosen so that texts with similar meaning
get similar vectors. That vector is called an **embedding**, and the model that
produces it is an **embedding model**.

```
"how do I work without wifi?"  ──embedding model──►  [0.021, -0.114, 0.078, ... ]   (384 numbers)
"using the app offline"        ──embedding model──►  [0.019, -0.101, 0.083, ... ]   (very close!)
"quarterly revenue report"     ──embedding model──►  [-0.201, 0.055, -0.142, ...]   (far away)
```

The number of values in the vector is its **dimensionality** (here, 384). Common
sizes are 384, 768, 1024, 1536, and 3072. Bigger vectors can capture more nuance
but cost more storage and compute — a real trade-off you get to make.

You never interpret the individual numbers; they are only meaningful *relative to
each other*. What matters is the geometry: **similar meaning → nearby vectors.**

## 2. Measuring closeness: cosine similarity

The standard way to score how similar two embeddings are is **cosine
similarity** — essentially the cosine of the angle between the two vectors:

- **1.0** — pointing the same direction (very similar meaning)
- **0.0** — perpendicular (unrelated)
- **−1.0** — opposite (rare with text embeddings)

Embedding models are trained to be used **normalized** (each vector scaled to
length exactly 1), and the labs normalize by default. When vectors are
normalized, cosine similarity is just their **dot product** — a cheap sum of
multiplications — which is why vector search is so fast.

> **Semantic vs. keyword.** Old-style keyword search (Lesson 5's BM25) matches
> *words*: "wifi" only finds documents containing "wifi." Semantic search matches
> *meaning*: "work without wifi" finds a doc titled "offline mode" even with zero
> shared words. That is the superpower — and, as we will see, keyword search
> still wins for exact things like error codes, so real systems use both.

## 3. Lab 1 — see meaning become geometry

Run [`labs/lab01_embeddings.py`](./labs/lab01_embeddings.py):

```bash
cd labs && source .venv/bin/activate
python lab01_embeddings.py
```

It does three things:

1. Embeds a handful of sentences and prints the **cosine-similarity matrix**, so
   you can see synonyms score high and unrelated sentences score low.
2. Runs a tiny **semantic search**: given a query, it ranks the sentences by
   similarity — and matches paraphrases that share no keywords.
3. Searches the real Nimbus corpus with a meaning-based query.

> **Note on offline mode.** If you have not installed the embedding model (or set
> `RAGKIT_OFFLINE=1`), the lab uses a lexical stand-in that matches shared
> *words*, not meaning. In that mode the synonym magic disappears and the numbers
> can even come out **backwards** (two sentences that merely share common words
> can outscore true synonyms). That is exactly the limitation of keyword matching
> this lesson is about — install `sentence-transformers` (it is in
> `requirements.txt`) to see real semantic search. The first run downloads the
> ~130 MB model once.

## 4. Choosing an embedding model

You have three broad options. **This course defaults to a free, local model** —
you never need the others, but you should know they exist.

**Free & local (what the labs use):** run an open model on your own machine with
`sentence-transformers`. Good small choices in 2026:

- **`BAAI/bge-small-en-v1.5`** — 384-dim, ~130 MB, fast on a laptop *(the default)*.
- **`BAAI/bge-base-en-v1.5`** — 768-dim, stronger, ~440 MB.
- **`nomic-ai/nomic-embed-text-v1.5`** — 768-dim, 8k-token context, great quality.
- **`Qwen/Qwen3-Embedding-0.6B`** — a top open model as of 2026; multilingual.

**Commercial APIs (optional, paid):** you send text and get vectors back. Often
higher quality, zero local compute, but a per-token cost and your text leaves
your machine.

- **Voyage AI** — the Voyage 4 family (Jan 2026); Anthropic officially recommends
  Voyage for use with Claude. Has code/finance/legal specialist models.
- **OpenAI** — `text-embedding-3-small` (1536-dim) / `-large` (3072-dim).
- **Cohere** — `embed-v4` (multimodal, up to 128k-token context).
- **Google** — `gemini-embedding-001` (3072-dim).

> **Anthropic makes no embedding model.** Claude is a *generation* model. For
> embeddings, Anthropic's docs point you to a third party such as Voyage AI. So
> even an all-Claude stack pairs Claude (generation) with a separate embedder.

### Terms you will meet on model cards

- **Max context / token limit** — the longest text one embedding call accepts
  (256 for old MiniLM, 8k for many, 32k+ for Voyage/Qwen, 128k for Cohere).
  Anything longer is silently truncated — which is *why we chunk* (Lesson 2).
- **Matryoshka / MRL** — some models let you *truncate* the vector (e.g. 768 → 256
  dims) and still get usable results, trading a little accuracy for less storage.
- **Multilingual vs. English-only** — English-only models fail on other
  languages; multilingual models embed many languages into one shared space.
- **Domain-specific** — specialist models (code, finance, legal) beat general
  ones inside their domain.

### The one rule that will bite you

**Embed your documents and your queries with the *same* model, and never mix
vectors from different models in one index.** Different models produce vectors in
different, incomparable spaces. If you change embedding models, you must
re-embed (re-index) everything. Treat the embedding model as part of your index's
identity.

## 5. Why this is not the whole story

Semantic search over raw documents has two gaps we spend the next lessons closing:

- Documents are often too long to embed as one vector (context limit) and too
  coarse to retrieve precisely. → **Chunking (Lesson 2).**
- Comparing a query against every vector is slow once you have millions. →
  **Vector databases and ANN indexes (Lesson 3).**

## Key takeaways

- An **embedding** turns text into a vector so that similar meaning → nearby
  vectors. Its length is its **dimensionality**.
- **Cosine similarity** (a dot product for normalized vectors) scores closeness;
  ranking by it *is* semantic search.
- Use the **same model** for documents and queries; changing models means
  re-embedding everything.
- The labs use a free local model (`bge-small-en-v1.5`); commercial APIs
  (Voyage, OpenAI, Cohere, Gemini) are optional upgrades. Anthropic has no
  embedding model — pair Claude with a separate embedder.
- A model's **max token limit** is why we chunk; **MRL** lets you shrink vectors.

## Further reading

- Sentence-Transformers docs — <https://www.sbert.net/>
- MTEB embedding leaderboard — <https://huggingface.co/spaces/mteb/leaderboard>
- Anthropic embeddings guidance — <https://docs.claude.com/en/docs/build-with-claude/embeddings>

Next: **[Lesson 2 — Chunking and ingestion](./02-chunking-and-ingestion.md)**.
