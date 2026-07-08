"""
Lab 5 — Better retrieval: dense + BM25, fused with RRF, then reranked.

For one query, it shows FOUR rankings side by side:
    1. dense only        (semantic embeddings)
    2. BM25 only         (exact keyword matching)
    3. hybrid            (dense + BM25 fused with Reciprocal Rank Fusion)
    4. hybrid + rerank   (a cross-encoder re-scores the top hybrid candidates)

Run it:   python lab05_hybrid_rerank.py

Note: the "BM25 rescues an exact term that semantics ranks too low" effect is
clearest with the REAL embedding model. In offline mode the dense stand-in is
also lexical, so dense and BM25 look more alike — but the RRF and rerank
MECHANICS still work and reorder the lists.
"""

import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

import ragkit

QUERY = "Why am I getting a 429 error from the API and how do I fix it?"
POOL = 10   # how many hybrid candidates to hand the reranker
SHOW = 5    # rows to print per ranking


def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def rrf(rankings, k=60):
    """
    Reciprocal Rank Fusion. `rankings` is a list of ranked id-lists (best first).
    Each id gets 1/(k+rank) from each list; we sum and re-sort. k=60 is the
    canonical default. Fuses RANKS, so incompatible score scales don't matter.
    """
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True), scores


# --- Build the two retrievers ---------------------------------------------
chunks = ragkit.chunk_corpus()
texts = [c["text"] for c in chunks]

embedder = ragkit.get_embedder()
chunk_vecs = embedder.encode(texts)
qv = embedder.encode([QUERY])[0]

bm25 = BM25Okapi([tokenize(t) for t in texts])

# Full rankings (best -> worst) from each retriever
dense_rank = [i for i, _ in ragkit.top_k(qv, chunk_vecs, k=len(chunks))]
bm25_scores = bm25.get_scores(tokenize(QUERY))
bm25_rank = list(np.argsort(bm25_scores)[::-1])

# Hybrid via RRF
hybrid_rank, _ = rrf([dense_rank, bm25_rank])

# Rerank the top POOL hybrid candidates with a cross-encoder
reranker = ragkit.get_reranker()
cand = hybrid_rank[:POOL]
rr_scores = reranker.scores(QUERY, [texts[i] for i in cand])
reranked = [i for i, _ in sorted(zip(cand, rr_scores), key=lambda p: p[1], reverse=True)]


def label(idx):
    snip = " ".join(texts[idx].split())[:52]
    return f"{chunks[idx]['id']:<22} {snip}"


# The GOLD chunk: the one that actually answers the query (the rate-limit / 429
# passage). We find it by content so we can track where each method ranks it.
gold = next(i for i, t in enumerate(texts) if "429" in t and "Retry-After" in t)

print(f"QUERY: {QUERY!r}")
print(f"GOLD chunk (the real answer): {chunks[gold]['id']}\n")
rankings = [
    ("1) dense only", dense_rank),
    ("2) BM25 only", bm25_rank),
    ("3) hybrid (RRF)", hybrid_rank),
    ("4) hybrid + rerank", reranked),
]
for name, ranking in rankings:
    print(name)
    for rank, idx in enumerate(ranking[:SHOW], 1):
        star = " <-- GOLD" if idx == gold else ""
        print(f"   {rank}. {label(idx)}{star}")
    print()

# --- Where did each method rank the GOLD (answer) chunk? -------------------
print("-" * 64)
print("Rank of the GOLD chunk in each method (lower = better; the reranked")
print("column includes only the top-%d hybrid candidates):" % POOL)
for name, ranking in rankings:
    pos = ranking.index(gold) + 1 if gold in ranking else None
    print(f"   {name:20s}: {pos if pos else 'not in candidate pool'}")
print("With the REAL embedding model + cross-encoder, hybrid and reranking pull")
print("the gold chunk up; offline (both stand-ins are lexical) they may not.")

# --- Show the RRF arithmetic for a chunk ranked DIFFERENTLY by the two -----
# Pick a chunk whose dense and BM25 ranks differ, so the fusion is illustrative.
winner = next(
    (i for i in hybrid_rank[:SHOW] if dense_rank.index(i) != bm25_rank.index(i)),
    hybrid_rank[0],
)
d_pos = dense_rank.index(winner) + 1
b_pos = bm25_rank.index(winner) + 1
print("\n" + "-" * 64)
print(f"How RRF scored {chunks[winner]['id']} (ranked differently by each retriever):")
print(f"   dense rank #{d_pos} -> 1/(60+{d_pos}) = {1/(60+d_pos):.5f}")
print(f"   BM25  rank #{b_pos} -> 1/(60+{b_pos}) = {1/(60+b_pos):.5f}")
print(f"   RRF score = {1/(60+d_pos) + 1/(60+b_pos):.5f}  (sum of the two)")
print("\nTakeaway: RRF fuses RANKS, so a chunk both retrievers rate highly rises")
print("even though cosine and BM25 scores are on totally different scales.")
