"""
Lab 8 — Contextual Retrieval, implemented and measured.

Builds TWO indexes over the corpus:
  - plain      : chunk text as-is
  - contextual : a short context blurb prepended to each chunk before embedding
and compares retrieval quality (recall@k, MRR) on the Lesson 7 eval set.

The context blurb is written by your local LLM if one is available; otherwise a
deterministic template (document title + source) is used so the lab still runs
and measures. Either way you see the mechanic: adding document context to a chunk
changes what it matches.

Run it:   python lab08_contextual.py
(With a real LLM this makes one short call per chunk — a bit slower, but that is
exactly what Contextual Retrieval does at ingestion time.)
"""

import json
import os

import numpy as np

import ragkit

K = 5
HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "data", "eval.json")) as f:
    eval_set = json.load(f)

chunks = ragkit.chunk_corpus()
embedder = ragkit.get_embedder()
llm = ragkit.get_llm()

CTX_SYSTEM = (
    "You write a single short sentence that situates a passage within its "
    "document, to improve search. Output only that sentence."
)


def make_context(chunk):
    """One-sentence context for a chunk. LLM if available, else a template."""
    if llm.name != "mock":
        prompt = (
            f"Document: {chunk['title']} ({chunk['source']})\n"
            f"Passage:\n{chunk['text'][:600]}\n\n"
            "Write one short sentence giving the context of this passage within "
            "the document (what it is about), to help a search engine find it."
        )
        try:
            return llm.generate(prompt, system=CTX_SYSTEM).strip().replace("\n", " ")
        except Exception:
            pass
    # deterministic fallback (still demonstrates + measures the mechanic)
    return f"This passage is from the '{chunk['title']}' page ({chunk['source']}) of the Nimbus Notes documentation."


# --- Build both variants ---------------------------------------------------
plain_texts = [c["text"] for c in chunks]
print(f"Writing context for {len(chunks)} chunks using: {llm.name} ...")
contexts = [make_context(c) for c in chunks]
ctx_texts = [f"{ctx}\n{c['text']}" for ctx, c in zip(contexts, chunks)]

plain_vecs = embedder.encode(plain_texts)
ctx_vecs = embedder.encode(ctx_texts)
sources = [c["source"] for c in chunks]


# --- Doc-level retrieval + metrics (same as Lesson 7) ----------------------
def ranked_docs(qv, vecs):
    seen, docs = set(), []
    for i, _ in ragkit.top_k(qv, vecs, k=len(chunks)):
        if sources[i] not in seen:
            seen.add(sources[i])
            docs.append(sources[i])
    return docs


def evaluate(vecs):
    recalls, mrrs = [], []
    for item in eval_set:
        qv = embedder.encode([item["question"]])[0]
        docs = ranked_docs(qv, vecs)[:K]
        rel = set(item["relevant_docs"])
        hits = [d for d in docs if d in rel]
        recalls.append(len(set(hits)) / len(rel) if rel else 0.0)
        mrr = 0.0
        for rank, d in enumerate(docs, 1):
            if d in rel:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)
    return np.mean(recalls), np.mean(mrrs)


p_recall, p_mrr = evaluate(plain_vecs)
c_recall, c_mrr = evaluate(ctx_vecs)

print("\nExample contextualized chunk:")
print(f"  context: {contexts[0]!r}")
print(f"  chunk  : {' '.join(chunks[0]['text'].split())[:70]}...\n")

print(f"{'index':12s} {'recall@'+str(K):>10s} {'MRR':>7s}")
print(f"{'plain':12s} {p_recall:10.2f} {p_mrr:7.2f}")
print(f"{'contextual':12s} {c_recall:10.2f} {c_mrr:7.2f}")

delta = (c_recall - p_recall, c_mrr - p_mrr)
print(f"\nDelta (contextual - plain): recall {delta[0]:+.2f}, MRR {delta[1]:+.2f}")
print("Contextual retrieval prepends document context so chunks that lost their")
print("meaning ('the plan costs 8 USD') regain it.")
if llm.name == "mock":
    print("\nNOTE: you are offline, so the 'context' is a fixed title template and the")
    print("embedder is lexical. That template adds the SAME words to every chunk of a")
    print("doc, so the delta here can be zero or even NEGATIVE — that is expected. The")
    print("real technique needs an LLM writing a specific blurb per chunk AND a semantic")
    print("embedder. Run with Ollama + sentence-transformers to see the intended gain.")
print("\nOn real corpora, with the LLM writing per-chunk context and hybrid+rerank on")
print("top, Anthropic reports up to ~67% fewer retrieval misses.")
