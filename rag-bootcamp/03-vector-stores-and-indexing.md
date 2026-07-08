# Lesson 3 — Vector databases and indexing

**Goal:** understand where embeddings live once you have millions of them, how
approximate search keeps queries fast, and how to use a real vector database
(Chroma) with metadata filtering.

---

## 1. Why not just loop over every vector?

In Lesson 1 we searched by computing the similarity of the query against *every*
stored vector and taking the top few. That **brute-force** (also called *exact*
or *flat*) search is perfectly correct and totally fine for a few thousand
chunks — it is what the labs have done so far.

But it is **O(N)**: double the corpus, double the work per query. At a million or
a hundred million chunks, scanning everything on every query is too slow. You
need a data structure that finds the nearest vectors *without* looking at all of
them. That is what a **vector database** provides.

## 2. Exact vs. Approximate Nearest Neighbor (ANN)

- **Exact / kNN** — check every vector; guaranteed to return the true nearest
  neighbors; slow at scale.
- **Approximate (ANN)** — use a clever index to check only a small, promising
  fraction of vectors; returns *almost* the true neighbors, far faster.

The quality of an ANN index is measured by **recall@k**: of the true top-k
neighbors, how many did it actually return? Production systems happily trade a
percent or two of recall for a 100× speedup — you tune where that trade sits.

## 3. Index types you will hear about

| Index | Idea | Trade-off |
| --- | --- | --- |
| **Flat** | No index — brute force | Exact, zero build cost, slow at scale. Great baseline. |
| **HNSW** | A navigable graph you "walk" toward the query | Excellent speed *and* recall, supports live inserts; uses more memory. **The default in most vector DBs.** |
| **IVF** | Cluster vectors; only search the nearest clusters | Low memory, fast build; recall depends on how many clusters you probe. Often paired with **PQ** compression at huge scale. |

You rarely implement these yourself — you pick a database and choose an index
type in its config. Knowing the names lets you read that config. **HNSW** is the
right default until memory or scale forces IVF/PQ.

Key HNSW knobs (same idea across databases): `M` (graph connectivity),
`ef_construction` (build-time thoroughness), `ef_search` (query-time
thoroughness — higher = better recall, slower).

## 4. Distance metrics

The database needs to know how to score "closeness":

- **Cosine** — angle only; the usual default for text embeddings.
- **Dot / inner product** — magnitude-sensitive; identical to cosine ranking
  *when vectors are normalized* (which most models do).
- **Euclidean (L2)** — straight-line distance.

Use whatever your embedding model was trained for. For normalized text
embeddings, cosine and dot product rank identically — pick cosine and move on.

## 5. Metadata filtering

A vector database stores structured **metadata** next to each vector, and lets
you filter on it *during* search: "find chunks about billing **where**
`team = 'finance'` **and** `year >= 2025`." This is essential for:

- **Freshness** — only search current documents.
- **Permissions** — never return a chunk the user isn't allowed to see. (Filter
  by access tags in the database; do not rely on the LLM to keep secrets.)
- **Precision** — narrow the search space before ranking.

Good databases apply the filter *inside* the ANN traversal so you don't
accidentally filter away all your candidates. You will use filtering in the lab.

## 6. The landscape (pick by your situation)

**For learning and prototyping (what the lab uses):**

- **Chroma** — an embedded, developer-friendly vector DB. `pip install chromadb`,
  a few lines to a working store, and `PersistentClient` saves to disk. The easy
  default in 2026.

**Also embedded / local:**

- **FAISS** — a *library*, not a database: the classic ANN engine. It can save
  and load an index, but it does not manage metadata or a database directory for
  you. Great when you want maximum control.
- **LanceDB** — embedded, disk-native, includes hybrid search.
- **sqlite-vec** — vector search inside SQLite; tiny and portable.

**Self-hosted or managed servers (production):**

- **pgvector** — vectors inside PostgreSQL; ideal if your data already lives
  there (pgvector 0.8 supports HNSW).
- **Qdrant**, **Weaviate**, **Milvus** — purpose-built engines with strong
  filtering and *native hybrid search*; self-host or cloud. Milvus targets the
  largest scales.
- **Pinecone** — fully managed, zero-ops (paid).

You can prototype on Chroma and move to Qdrant/pgvector/Pinecone later; the RAG
code barely changes because the concepts (upsert vectors + metadata, query
top-k, filter) are the same everywhere.

> **Hybrid search preview.** Note which databases do *native hybrid* (dense +
> keyword) search: Qdrant, Weaviate, Milvus, LanceDB, Pinecone. Chroma and plain
> pgvector do not, so with those you combine keyword search yourself — which is
> exactly what Lesson 5 teaches.

## 7. Lab 3 — build a real vector store with Chroma

[`labs/lab03_vector_store.py`](./labs/lab03_vector_store.py):

1. Chunks the corpus and **embeds** it.
2. Stores chunks + embeddings + **metadata** in a **persistent** Chroma
   collection (saved to disk, reloads next run).
3. Runs a semantic **query**, then the same query with a **metadata filter**.
4. Shows an **upsert** (updating a chunk) and confirms brute-force and Chroma
   return the same top result — the database is a faster path to the same math.

```bash
cd labs && source .venv/bin/activate
python lab03_vector_store.py
```

We pass our *own* embeddings to Chroma (rather than letting it download its own
model), which keeps the lab fast, offline-friendly, and makes the embed step
explicit.

## Key takeaways

- Brute-force search is fine small; **vector databases** use **ANN** indexes to
  stay fast at scale, trading a little **recall** for big speedups.
- **HNSW** is the default index; **IVF/PQ** for extreme scale; **flat** is the
  exact baseline.
- Use **cosine** for normalized text embeddings.
- **Metadata filtering** enforces freshness and permissions and sharpens
  precision — filter in the database, don't trust the LLM.
- **Chroma** is the easy local default; the same code ideas port to
  Qdrant/pgvector/Pinecone in production.

## Further reading

- Chroma docs — <https://docs.trychroma.com/>
- HNSW explained — <https://www.pinecone.io/learn/series/faiss/hnsw/>

Next: **[Lesson 4 — Your first RAG pipeline](./04-first-rag-pipeline.md)**.
