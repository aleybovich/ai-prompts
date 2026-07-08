# Lesson 7 — Evaluating RAG

**Goal:** stop guessing whether your RAG system is good. Build an evaluation set,
measure retrieval and answer quality with real metrics, and learn to fix the
right half.

"It seems to work" is how RAG projects die. Every improvement in Lessons 5–6 is a
knob; evaluation is how you know which way to turn it.

---

## 1. Two things to measure, in order

RAG has two failure modes, and they need different metrics:

1. **Retrieval quality** — did we fetch the chunks that contain the answer?
2. **Answer quality** — given the chunks, did the model produce a correct,
   grounded answer?

**Measure retrieval first.** If retrieval didn't surface the answer, no prompt
tweak can save the generation — you would be tuning the wrong half. Fix retrieval
until the right chunks reliably come back, *then* work on the answer.

## 2. Build an evaluation set

You need a small set of **questions** paired with ground truth. A practical eval
item has:

- the **question**,
- the **relevant document(s)** that contain the answer (for retrieval metrics),
- a **reference answer** or key facts the answer must contain (for answer
  metrics).

Twenty to fifty good items beat none by a mile. Write them from real user
questions, known tricky cases, and at least a few whose answer is **not** in the
corpus (to test abstention). The lab uses `labs/data/eval.json` — open it to see
the shape.

> Building the set is the work. You can bootstrap it: have an LLM draft candidate
> questions from your documents, then a human curates. But a human must own the
> ground truth, or you are grading against the same guesses you are trying to
> test.

## 3. Retrieval metrics

Run each eval question through retrieval, look at the top-k results, and compare
the documents you got against the ground-truth relevant ones. The standard
metrics (all reported as an average over your eval set):

| Metric | Question it answers | Formula (per query) |
| --- | --- | --- |
| **Hit rate @k** | Did *any* relevant doc make the top-k? | 1 if ≥1 relevant in top-k, else 0 |
| **Recall @k** | What fraction of the relevant docs did we get? | (relevant retrieved) / (total relevant) |
| **Precision @k** | How much of what we retrieved was relevant? | (relevant retrieved) / k |
| **MRR** | How high was the *first* relevant doc? | 1 / (rank of first relevant) |
| **NDCG @k** | Are the most relevant docs ranked highest? | position-discounted, normalized to ideal |

For most RAG work, **recall@k** and **MRR** are the two to watch: recall tells you
whether the evidence is present at all, MRR tells you whether it is near the top
where the model will actually use it.

## 4. Answer metrics

Once retrieval is solid, grade the generated answers. Three questions matter:

- **Faithfulness / groundedness** — is every claim supported by the retrieved
  context (vs. hallucinated)? The most important RAG-specific metric.
- **Answer relevance** — does the answer actually address the question?
- **Answer correctness** — does it match the reference answer / key facts?

How to measure them cheaply:

- **Key-fact check** — does the answer contain the required fact ("$8", "200
  MB")? Crude but zero-cost and catches gross failures. The lab does this.
- **LLM-as-judge** — a second LLM call scores faithfulness/relevance against a
  rubric. This is how Ragas, DeepEval, and TruLens work under the hood. Powerful,
  but mind the **biases**: judges favor longer answers, the first option in a
  pair, and their own outputs. Randomize order, calibrate against a few human
  labels, and spot-check.

## 5. Frameworks you'll meet (you don't need them to start)

The lab computes metrics by hand so you understand them. In production you'll
likely reach for a library:

- **Ragas** (v0.4.x) — the common RAG-eval library: faithfulness, answer
  relevancy, context precision, context recall.
- **DeepEval** — pytest-native, 50+ metrics, G-Eval (LLM-as-judge with reasoning).
- **TruLens** — the "RAG triad" (context relevance, groundedness, answer
  relevance) as feedback functions.
- **Phoenix / Arize** — open-source tracing + evaluation; great for *seeing* every
  retrieval and LLM step when debugging.

Start with hand-rolled metrics on a small set; adopt a framework when you want
LLM-judged faithfulness and tracing at scale.

## 6. The improvement loop

```
1. Run eval  ->  2. Read the FAILURES (not just the average)  ->  3. Diagnose:
   retrieval miss?  -> fix chunking / hybrid / rerank (Lessons 2,5)
   good chunks, bad answer? -> fix prompt / model (Lesson 6)
   -> 4. Change ONE thing  ->  5. Re-run eval, compare  ->  repeat
```

The averages tell you *if* you improved; the individual failures tell you *what*
to fix. Always read the failures.

## 7. Lab 7 — measure your retriever

[`labs/lab07_eval.py`](./labs/lab07_eval.py) loads the evaluation set, runs
retrieval for every question, and computes **hit@k, recall@k, precision@k, and
MRR** — implemented from scratch so the formulas are concrete. It reports metrics
for **dense** vs **hybrid** retrieval so you can see, in numbers, the improvement
Lesson 5 promised. It also runs the cheap **key-fact** answer check.

```bash
cd labs && source .venv/bin/activate
python lab07_eval.py
```

Offline (lexical) the absolute numbers are low; install the real embedding model
to see them jump — and to see hybrid beat dense. The point of the lab is the
*measurement machinery*, which is identical either way.

## Key takeaways

- Measure **retrieval first** (recall@k, MRR), then **answers** (faithfulness,
  relevance, correctness).
- Build a small **eval set** with questions, relevant docs, and reference facts —
  including unanswerable ones.
- **LLM-as-judge** powers faithfulness scoring but is biased — randomize,
  calibrate, spot-check.
- Use **Ragas / DeepEval / TruLens / Phoenix** in production; hand-rolled metrics
  to learn.
- Improve in a loop: **read the failures**, change one thing, re-measure.

## Further reading

- Ragas metrics — <https://docs.ragas.io/en/stable/concepts/metrics/>
- A survey of LLM-as-judge — search "Judging LLM-as-a-Judge" (Zheng et al., 2023).

Next: **[Lesson 8 — Advanced patterns](./08-advanced-patterns.md)**.
