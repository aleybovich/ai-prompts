"""
Lab 7 — Evaluate retrieval (and answer quality), metrics from scratch.

Loads data/eval.json (questions + ground-truth relevant docs + key facts),
runs DENSE and HYBRID retrieval for every question, and computes:
    hit@k, recall@k, precision@k, MRR
implemented by hand so the formulas are concrete. Then two answer-side checks:
    - EVIDENCE COVERAGE (offline): did retrieval put the required fact in the
      context at all? A precondition for a faithful answer.
    - KEY-FACT check (needs a real LLM): does the generated answer contain the
      required fact? A cheap correctness proxy.

Run it:   python lab07_eval.py

We use k=3 (the corpus is small — with 10 docs, k=5 would return half of them
and every metric saturates). The MEASUREMENT machinery is what matters.
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
from rank_bm25 import BM25Okapi

import ragkit

K = 3
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


def ranked_chunks(question, mode):
    """Ranked chunk indices (best first) for a retrieval mode."""
    qv = embedder.encode([question])[0]
    dense = [i for i, _ in ragkit.top_k(qv, chunk_vecs, k=len(chunks))]
    if mode == "dense":
        return dense
    bm = list(np.argsort(bm25.get_scores(re.findall(r"[a-z0-9_]+", question.lower())))[::-1])
    return rrf([dense, bm])


def ranked_docs(question, mode):
    """Collapse ranked chunks to a ranked list of UNIQUE source docs."""
    seen, docs = set(), []
    for idx in ranked_chunks(question, mode):
        if sources[idx] not in seen:
            seen.add(sources[idx])
            docs.append(sources[idx])
    return docs


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


# --- Retrieval metrics: dense vs hybrid -----------------------------------
print(f"Evaluating {len(eval_set)} questions, top-k = {K}\n")
print(f"{'mode':7s} {'hit@k':>7s} {'recall@k':>9s} {'prec@k':>7s} {'MRR':>6s}")
for mode in ["dense", "hybrid"]:
    agg = np.zeros(4)
    for item in eval_set:
        agg += metrics_for(ranked_docs(item["question"], mode), item["relevant_docs"])
    agg /= len(eval_set)
    print(f"{mode:7s} {agg[0]:7.2f} {agg[1]:9.2f} {agg[2]:7.2f} {agg[3]:6.2f}")
print("\n(precision@k looks low because most questions have a single relevant doc,")
print(" so the best possible precision@3 is 1/3. Watch recall@k and MRR instead.")
print(" NDCG@k is omitted here: relevance is binary and doc-level, so it would")
print(" collapse toward MRR — it shines when you have graded relevance labels.)")

# --- Read the FAILURES, not just the average ------------------------------
print("\nPer-question (hybrid) — read the misses:")
for item in eval_set:
    docs = ranked_docs(item["question"], "hybrid")
    hit, recall, _, mrr = metrics_for(docs, item["relevant_docs"])
    if not hit or mrr < 1.0:  # show anything less than perfect
        mark = "MISS" if not hit else "low "
        print(f"  [{mark}] {item['id']}: got {docs[:3]} | want {item['relevant_docs']} | MRR={mrr:.2f}")


def contains_fact(text, fact):
    """Word-boundary match so '8' doesn't match '80' and 'Pro' not 'Profile'."""
    return re.search(r"\b" + re.escape(fact) + r"\b", text, re.IGNORECASE) is not None


# --- Evidence coverage (faithfulness precursor, runs offline) --------------
print("\nEvidence coverage: is the required fact even in the retrieved context?")
covered = 0
for item in eval_set:
    idxs = ranked_chunks(item["question"], "hybrid")[: K + 2]
    ctx = " ".join(texts[i] for i in idxs)
    ok = all(contains_fact(ctx, kw) for kw in item["must_include"])
    covered += ok
print(f"  {covered}/{len(eval_set)} questions have all key facts in context.")
print("  (If the fact isn't retrieved, no prompt can make the answer faithful —")
print("   fix retrieval first.)")

# --- Key-fact answer check (needs a real LLM) -----------------------------
llm = ragkit.get_llm()
if llm.name == "mock":
    print("\n(Answer-generation check skipped: no real LLM. `ollama pull gemma3:1b`.)")
else:
    print("\nAnswer key-fact check (does the generated answer contain the fact?):")
    SYSTEM = "Answer using ONLY the context. If not present, say you don't know. Be concise."
    passed = 0
    for item in eval_set:
        idxs = ranked_chunks(item["question"], "hybrid")[:K]     # ranked order
        ctx = "\n\n".join(f"[{sources[i]}] {texts[i]}" for i in idxs)
        ans = llm.generate(f"Context:\n{ctx}\n\nQuestion: {item['question']}\n\nAnswer:", system=SYSTEM)
        ok = all(contains_fact(ans, kw) for kw in item["must_include"])
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {item['id']}: needs {item['must_include']}")
    print(f"\n  Key-fact pass rate: {passed}/{len(eval_set)}")
