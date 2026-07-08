"""
Lab 7 — Evaluate retrieval (and a cheap answer check), metrics from scratch.

Loads data/eval.json (questions + ground-truth relevant docs + key facts),
runs DENSE and HYBRID retrieval for every question, and computes:
    hit@k, recall@k, precision@k, MRR
implemented by hand so the formulas are concrete. Then a cheap key-fact answer
check via the LLM (skipped cleanly if only the MockLLM is available).

Run it:   python lab07_eval.py

Offline (lexical) the numbers are low and dense ~ hybrid; install the real
embedding model to watch scores jump and hybrid beat dense. The MEASUREMENT
machinery is identical either way.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

import ragkit

K = 5
HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "data", "eval.json")) as f:
    eval_set = json.load(f)

chunks = ragkit.chunk_corpus()
texts = [c["text"] for c in chunks]
sources = [c["source"] for c in chunks]

embedder = ragkit.get_embedder()
chunk_vecs = embedder.encode(texts)
bm25 = BM25Okapi([re.findall(r"[a-z0-9_]+", t.lower()) for t in texts])


def rrf(rankings, k=60):
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] += 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def retrieve_docs(question, mode):
    """Return a ranked list of UNIQUE source docs (best first)."""
    qv = embedder.encode([question])[0]
    dense = [i for i, _ in ragkit.top_k(qv, chunk_vecs, k=len(chunks))]
    if mode == "dense":
        order = dense
    else:  # hybrid
        bm = list(np.argsort(bm25.get_scores(re.findall(r"[a-z0-9_]+", question.lower())))[::-1])
        order = rrf([dense, bm])
    seen, docs = set(), []
    for idx in order:
        if sources[idx] not in seen:
            seen.add(sources[idx])
            docs.append(sources[idx])
    return docs


# --- Retrieval metrics (per-query, then averaged) -------------------------
def metrics_for(retrieved_docs, relevant_docs, k=K):
    topk = retrieved_docs[:k]
    rel = set(relevant_docs)
    hits = [d for d in topk if d in rel]
    hit = 1.0 if hits else 0.0
    recall = len(set(hits)) / len(rel) if rel else 0.0
    precision = len(hits) / k
    mrr = 0.0
    for rank, d in enumerate(topk, start=1):
        if d in rel:
            mrr = 1.0 / rank
            break
    return hit, recall, precision, mrr


print(f"Evaluating {len(eval_set)} questions, top-k = {K}\n")
print(f"{'mode':7s} {'hit@k':>7s} {'recall@k':>9s} {'prec@k':>7s} {'MRR':>6s}")
for mode in ["dense", "hybrid"]:
    agg = np.zeros(4)
    for item in eval_set:
        docs = retrieve_docs(item["question"], mode)
        agg += metrics_for(docs, item["relevant_docs"])
    agg /= len(eval_set)
    print(f"{mode:7s} {agg[0]:7.2f} {agg[1]:9.2f} {agg[2]:7.2f} {agg[3]:6.2f}")

# --- Show a couple of per-question results so failures are visible ---------
print("\nPer-question (hybrid) — read the failures, not just the average:")
for item in eval_set[:6]:
    docs = retrieve_docs(item["question"], "hybrid")
    hit, recall, _, mrr = metrics_for(docs, item["relevant_docs"])
    mark = "OK " if hit else "MISS"
    print(f"  [{mark}] {item['id']}: got {docs[:3]} | want {item['relevant_docs']} | MRR={mrr:.2f}")

# --- Cheap answer check (key facts) ---------------------------------------
llm = ragkit.get_llm()
if llm.name == "mock":
    print("\n(Answer check skipped: no real LLM. Install Ollama to grade answers.)")
else:
    print("\nAnswer key-fact check (does the answer contain the required fact?):")
    passed = 0
    SYSTEM = "Answer using ONLY the context. If not present, say you don't know. Be concise."
    for item in eval_set:
        docs_ranked = retrieve_docs(item["question"], "hybrid")
        # gather the text of top chunks whose source is in the top docs
        top_docs = set(docs_ranked[:3])
        ctx = "\n\n".join(f"[{sources[i]}] {texts[i]}" for i in range(len(texts)) if sources[i] in top_docs)[:4000]
        ans = llm.generate(f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\nAnswer:", system=SYSTEM)
        ok = all(kw.lower() in ans.lower() for kw in item["must_include"])
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item['id']}: needs {item['must_include']}")
    print(f"\n  Key-fact pass rate: {passed}/{len(eval_set)}")
