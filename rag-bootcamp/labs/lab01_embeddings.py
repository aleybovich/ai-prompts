"""
Lab 1 — Embeddings and semantic search.

You will see:
  1. A cosine-similarity MATRIX: synonyms score high, unrelated text scores low.
  2. A tiny semantic search that matches meaning, not just shared words.
  3. The same idea run over the real Nimbus Notes corpus.

Run it:   python lab01_embeddings.py

Tip: install sentence-transformers (in requirements.txt) for real semantic
results. With RAGKIT_OFFLINE=1 a lexical stand-in is used and synonyms won't pop.
"""

import numpy as np

import ragkit

embedder = ragkit.get_embedder()

# --- 1. Cosine-similarity matrix ------------------------------------------
sentences = [
    "How do I use the app without an internet connection?",  # 0
    "Working offline with no wifi",                          # 1  (synonym of 0)
    "What does the Pro plan cost?",                          # 2
    "Pricing for the paid subscription tier",                # 3  (synonym of 2)
    "The mitochondria is the powerhouse of the cell",        # 4  (unrelated)
]

vecs = embedder.encode(sentences)          # shape: (5, dim), normalized
sim = vecs @ vecs.T                        # cosine similarity (dot of normalized vectors)

print("Cosine-similarity matrix (higher = more similar):\n")
print("      " + "  ".join(f"s{j}" for j in range(len(sentences))))
for i, row in enumerate(sim):
    print(f"s{i}:  " + "  ".join(f"{v:+.2f}" for v in row))
print()
for i, s in enumerate(sentences):
    print(f"  s{i}: {s}")

print(f"\n  s0 vs s1 (offline synonyms): {sim[0,1]:+.2f}")
print(f"  s0 vs s4 (unrelated)       : {sim[0,4]:+.2f}")
print("  -> With a real model, the synonym pair scores much higher than the")
print("     unrelated pair, even though the sentences share few/no words.")

# --- 2. Tiny semantic search ----------------------------------------------
print("\n" + "-" * 60)
print("Semantic search: rank the sentences by similarity to a query\n")
query = "cancel my paid plan"
q_vec = embedder.encode([query])[0]
for rank, (idx, score) in enumerate(ragkit.top_k(q_vec, vecs, k=3), 1):
    print(f"  {rank}. ({score:+.2f}) {sentences[idx]}")
print(f"\n  Query was: {query!r}")
print("  A real model ranks the pricing/subscription sentences at the top,")
print("  matching by MEANING even though 'cancel' appears in none of them.")

# --- 3. Search the real corpus --------------------------------------------
print("\n" + "-" * 60)
print("Search the Nimbus corpus (whole documents) by meaning\n")
docs = ragkit.load_corpus()
doc_vecs = embedder.encode([d["text"] for d in docs])
corpus_query = "keep my notes safe and private"
qv = embedder.encode([corpus_query])[0]
print(f"  Query: {corpus_query!r}\n")
for rank, (idx, score) in enumerate(ragkit.top_k(qv, doc_vecs, k=3), 1):
    print(f"  {rank}. ({score:+.2f}) {docs[idx]['title']}  [{docs[idx]['id']}]")
print("\n  The security/encryption doc should rank near the top — again, by")
print("  meaning, not by keyword overlap.")
