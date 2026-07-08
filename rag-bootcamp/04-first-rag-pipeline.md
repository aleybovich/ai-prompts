# Lesson 4 — Your first RAG pipeline

**Goal:** connect Lessons 1–3 into one working system: **retrieve → augment →
generate**, wrapped in a function you can call with any question.

You already saw a teaser in Lab 0. Now we build the real thing — an index you
create once and a query function you can reuse — and look closely at the prompt,
because the prompt is where retrieval and generation meet.

---

## 1. The two halves, together

```
INGESTION (once):   load -> chunk -> embed -> store in Chroma        [Lessons 2,3]
                                                     │
QUERY (per question):                                ▼
   question -> embed -> search top-k -> build prompt -> LLM -> answer + sources
                        [Lesson 1,3]     (this lesson)   [Lesson 0]
```

Everything to the left of "build prompt" is retrieval. The two new pieces in this
lesson are **assembling the prompt** and **returning sources with the answer**.

## 2. Choosing top-k

**top-k** is how many chunks you retrieve and put in the prompt. It is a
balance:

- **Too few** (k=1–2): you might miss the chunk that holds the answer, or a
  question that needs two facts (e.g. "which plans have SSO *and* what does it
  cost?") only gets half its evidence.
- **Too many** (k=20): you spend tokens, add latency, and dilute the prompt with
  irrelevant text — which can *lower* answer quality and invite the model to use
  the wrong passage.

Start with **k = 3–5** for a first system. Later lessons add reranking (Lesson 5)
so you can retrieve a wider net cheaply and then keep only the best few.

## 3. The RAG prompt template

A grounded RAG prompt has three parts, in this order:

1. **A system instruction** that sets the rules: answer *only* from the provided
   context; if the answer isn't there, say so; cite sources.
2. **The retrieved context**, clearly delimited, each chunk labeled with its
   source so the model (and you) can cite it.
3. **The user's question.**

```
SYSTEM: You are a support assistant. Answer using ONLY the context below.
        If the answer is not in the context, say you don't know. Cite the
        [source] for each fact.

CONTEXT:
[pricing.md] The Pro plan costs 8 USD per month...
[security.md] Team plans add single sign-on (SSO)...

QUESTION: Which plan has SSO and how much is it?
```

Why this shape works: putting the rules first *primes* the model, delimiting the
context stops it from confusing instructions with data, and labeling sources
makes citations (Lesson 6) trivial. Lesson 6 goes deeper on grounding; this
lesson establishes the pattern.

> **In code**, the system instruction is passed as a separate **`system` role**
> rather than concatenated into the context/question string — that is why the lab
> prints the `SYSTEM:` block apart from the `PROMPT:` block. Keeping rules in the
> system role separates instructions from data cleanly.

## 4. Return sources, always

A RAG answer should never be just text. Return the **sources** (which chunks were
used) alongside it. This gives you:

- **Trust** — users can verify the answer against the real document.
- **Debuggability** — when an answer is wrong, you immediately see whether
  retrieval fetched the wrong chunks (a retrieval bug) or the model misused good
  chunks (a generation bug). That distinction drives every fix you will make.

## 5. Lab 4 — a reusable RAG function

[`labs/lab04_first_rag.py`](./labs/lab04_first_rag.py) builds a persistent Chroma
index once, then defines a single `answer(question)` function that retrieves,
builds the prompt, calls the LLM, and returns the answer **plus its sources**. It
runs that function over several real questions.

```bash
cd labs && source .venv/bin/activate
python lab04_first_rag.py
```

With **no LLM installed**, you still see retrieval working and the exact grounded
prompt for each question (via the MockLLM). Install Ollama (`ollama pull
gemma3:1b`) to get real answers.

> **Offline scores.** In offline mode the printed `sim` numbers reflect
> word-overlap, not meaning — don't read into their magnitude (an unanswerable
> question can even show a high score). Install the real embedder for meaningful
> similarity scores. Try editing the questions at the bottom of the
file — including one whose answer is *not* in the docs, to see whether the system
correctly says "I don't know" (a preview of Lesson 6's honesty problem).

## 6. What can go wrong (and which lesson fixes it)

Your first pipeline will have failure modes. Naming them maps the rest of the
course:

| Symptom | Likely cause | Fixed in |
| --- | --- | --- |
| Right doc exists but wasn't retrieved | Weak retrieval / query wording | Lesson 5 (hybrid, rerank, query rewrite) |
| Exact term (error code, plan name) missed | Semantic search ignores exact tokens | Lesson 5 (BM25 hybrid) |
| Retrieved good chunks, answer still wrong | Weak prompt / model ungrounded | Lesson 6 (grounding, citations) |
| "Is it actually any good?" | No measurement | Lesson 7 (evaluation) |

## Key takeaways

- A RAG pipeline is **ingest once**, then **retrieve → augment → generate** per
  question.
- **top-k = 3–5** is a sane start; reranking later lets you cast a wider net.
- The **prompt template** is: system rules → delimited, source-labeled context →
  question.
- Always **return sources** with the answer — it is the difference between a demo
  and a debuggable system.

Next: **[Lesson 5 — Better retrieval](./05-better-retrieval.md)**.
