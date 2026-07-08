# Lesson 5 — Better retrieval: hybrid search, reranking, query rewriting

**Goal:** fix the most common reason RAG answers are wrong — retrieval missed the
right chunk. You will add keyword search, fuse it with semantic search, rerank
the results with a precise model, and reshape the query itself.

Retrieval quality is the ceiling on answer quality. This is the lesson that
raises the ceiling.

---

## 1. Where plain semantic search fails

Semantic (dense) search is great at meaning but has real blind spots:

- **Exact tokens.** Ask about error `NIMBUS_RATE_LIMIT_429` or the "Team" plan and
  embeddings may rank a vaguely-related paragraph above the one with the exact
  string. Embeddings smear specifics into "general topic."
- **Rare words / proper nouns / codes.** These barely move a semantic vector but
  are exactly what the user typed.
- **Query phrasing.** A terse or oddly-worded query embeds to the wrong
  neighborhood.

Each has a fix. Stack them.

## 2. Hybrid search: dense + keyword, fused with RRF

**BM25** is classic keyword ranking (an evolved TF-IDF). It nails exact matches —
codes, names, error strings — but understands no meaning. Dense embeddings
understand meaning but fumble exact tokens. They **fail in opposite ways**, so
combining them catches what either misses. Running both and merging is **hybrid
search**, the production default.

The catch: you cannot just add the scores. BM25 scores are unbounded positive
numbers; cosine similarities live in [−1, 1]. Averaging them is meaningless.

**Reciprocal Rank Fusion (RRF)** solves this elegantly by fusing **ranks, not
scores**. For each result, from each retriever, add `1 / (k + rank)` (rank starts
at 1; the constant **k = 60** is the well-known default). Sum across retrievers;
sort by the summed score.

```
RRF_score(doc) = Σ over retrievers of  1 / (k + rank_of_doc_in_that_retriever)
```

A document ranked #1 by BM25 and #3 by dense beats one ranked #10 by both.
Scores never need to be comparable — only ranks — which is why RRF is robust and
everywhere.

## 3. Reranking: retrieve wide, then be picky

First-stage retrieval (dense or hybrid) is *fast* but *approximate* — it scores
the query and each chunk **separately** (a "bi-encoder"). A **reranker** is a
**cross-encoder**: it reads the query and one candidate chunk **together** and
scores how well that chunk actually answers the query. Far more accurate — and
far too slow to run over the whole corpus.

So use the two-stage pattern:

```
retrieve TOP ~25 (cheap, hybrid)  ──►  rerank to TOP ~4 (accurate cross-encoder)  ──►  LLM
```

Cast a wide, cheap net; let the precise-but-slow model pick the final few. This
single step is often the biggest quality win in a RAG system.

**Reranker options in 2026:**

- **Free & local (labs):** a cross-encoder like `ms-marco-MiniLM-L-6-v2` (~80 MB)
  via `sentence-transformers`. Runs on a laptop.
- **Commercial APIs:** **Cohere Rerank 4** (Dec 2025; 32k-token context, 100+
  languages) and **Voyage rerank-2.5** are strong hosted rerankers.

## 4. Query transformation: fix the question before you search

Sometimes the problem is the query, not the index. Rewrite it *before* retrieval:

- **Query rewriting** — clean up a messy or conversational query, or turn a
  follow-up ("what about the Team plan?") into a standalone question.
- **Multi-query** — have an LLM produce 3–4 paraphrases, retrieve for each, and
  merge (RRF again). Raises recall when one phrasing misses.
- **HyDE (Hypothetical Document Embeddings)** — have the LLM write a *fake answer*
  to the question, then embed and search with *that*. Answer-shaped text often
  sits closer to real answer chunks than the question does. Best for short or
  under-specified queries over descriptive corpora.
- **Decomposition** — split a complex question ("compare X and Y") into
  sub-questions, retrieve for each, then synthesize. The backbone of multi-hop
  retrieval.

These cost an extra LLM call or two, so reach for them when evaluation (Lesson 7)
shows retrieval — not generation — is the bottleneck.

> **Multi-turn (chat) RAG — the most common real case.** Most RAG ships as a
> chatbot, where the user's message depends on the conversation: "What about the
> Team plan?" or "How much is that?" You **cannot embed that directly** — "that"
> and "it" mean nothing to the retriever. The standard fix is **query
> condensation**: before retrieving, make one cheap LLM call that rewrites the
> latest message *plus the recent history* into a **standalone question** ("How
> much does the Team plan cost per user?"), then run your normal retrieval on the
> rewritten query. This one step is what makes conversational RAG actually work —
> add it whenever your system has a back-and-forth interface.

## 5. MMR: relevant *and* diverse

If your top results are five near-duplicate chunks, the LLM sees one fact five
times and misses others. **Maximal Marginal Relevance (MMR)** picks each next
result to balance *relevance to the query* against *dissimilarity to what you
already picked*, tuned by a knob λ (λ=1 pure relevance, λ=0 pure diversity).
Useful when a corpus repeats itself; skip it when every chunk is distinct.

## 6. Putting the retrieval stack together

A strong retrieval pipeline, in order:

```
query
  │  (optional) rewrite / multi-query / HyDE            [§4]
  ▼
dense search ─┐
              ├─ RRF fuse ─► top ~25 candidates          [§2]
BM25 search ──┘
  │  (optional) metadata filter                          [Lesson 3]
  ▼
cross-encoder rerank ─► top ~4                            [§3]
  │  (optional) MMR for diversity                        [§5]
  ▼
top-k chunks → prompt → LLM                              [Lesson 4]
```

You will not always need every stage. Start with **hybrid + rerank** — it fixes
most retrieval failures — and add query transforms or MMR only when measurement
says so.

## 7. Lab 5 — hybrid + RRF + rerank, side by side

[`labs/lab05_hybrid_rerank.py`](./labs/lab05_hybrid_rerank.py) builds BM25 and
dense retrievers over the corpus and, for a query about an API rate-limit error,
shows four rankings side by side: **dense only**, **BM25 only**, **hybrid
(RRF)**, and **hybrid + rerank**. It marks the *gold* chunk (the one that really
answers the query) and prints its rank under each method, so you can watch
keyword search and reranking pull it upward — with the **real** models.

```bash
cd labs && source .venv/bin/activate
python lab05_hybrid_rerank.py
```

The lab implements RRF from scratch (it is ~10 lines) so the fusion is not
magic. The cross-encoder reranker downloads ~80 MB on first use.

> **Offline caveat.** With `RAGKIT_OFFLINE=1`, *both* the dense retriever and the
> reranker fall back to lexical stand-ins, so dense looks a lot like BM25 and the
> reranker may reorder *worse*, not better. The RRF and rerank **mechanics** run
> and reorder the lists, but the quality gains — BM25 rescuing an exact term,
> the cross-encoder sharpening the top — only appear with the real models
> (install `sentence-transformers`). The lab prints which backends it used.

## Key takeaways

- Retrieval failure is the #1 cause of wrong RAG answers; this lesson is the
  highest-leverage fix.
- **Hybrid search** = dense (meaning) + **BM25** (exact terms), fused with **RRF**
  (merge ranks, k=60) — the production default.
- **Rerank** with a cross-encoder: retrieve ~25 cheaply, keep the best ~4. Often
  the single biggest quality gain.
- **Query transformation** (rewrite, multi-query, HyDE, decomposition) fixes bad
  *questions*; **MMR** fixes redundant *results*.
- Add stages guided by evaluation, not vibes — start with hybrid + rerank.

## Further reading

- Reciprocal Rank Fusion (Cormack et al., 2009) — the original RRF paper.
- HyDE (Gao et al., 2022) — <https://arxiv.org/abs/2212.10496>
- Cohere Rerank — <https://docs.cohere.com/docs/rerank>

Next: **[Lesson 6 — Grounded generation and citations](./06-generation-and-citations.md)**.
