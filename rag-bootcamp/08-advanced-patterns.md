# Lesson 8 — Advanced patterns

**Goal:** know the techniques beyond the core pipeline — contextual retrieval,
agentic RAG, and GraphRAG — well enough to recognize when a problem needs one,
and to decide when a framework earns its place.

You can ship a great system with Lessons 1–7. These patterns solve *specific*
harder problems. Reach for them when evaluation says the basics have topped out.

---

## 1. Contextual Retrieval (fix chunks losing their context)

**The problem.** Chunking strips a chunk of its surroundings. A chunk that reads
"The plan costs 8 USD per month" no longer says *which* plan — so a query about
the "Pro plan price" may not match it well, and the model may misread it.

**The fix (Anthropic's Contextual Retrieval).** Before embedding each chunk, use
a cheap LLM to write a one- or two-sentence **context blurb** that situates the
chunk in its document, and prepend it:

```
Original chunk:  "The plan costs 8 USD per month, or 80 USD per year..."
Contextualized:  "This chunk is from the Nimbus Notes pricing page, describing
                  the Pro plan. The plan costs 8 USD per month, or 80 USD..."
```

Then embed *and* BM25-index the contextualized chunk. Two parts:

1. **Contextual Embeddings** — embed chunk-with-context.
2. **Contextual BM25** — keyword-index chunk-with-context too.

**Reported results** (Anthropic, on their eval — vendor-reported): contextual
embeddings cut top-20 retrieval failures by **~35%**; adding contextual BM25,
**~49%**; adding **reranking** on top, **~67%**. It stacks with everything you
already learned.

**Cost.** Writing a blurb per chunk sounds expensive, but **prompt caching**
(Lesson 9) makes it cheap: cache the whole document once, then each chunk only
pays for its short blurb. Anthropic's example put this near ~$1 per million
document tokens. It is an **ingestion-side** cost you pay once.

The lab implements this and measures the difference. (Related, from Lesson 2:
**late chunking** attacks the same "lost context" problem on the embedding side,
by embedding the whole document first and splitting afterward.)

## 2. Agentic RAG (let the model drive retrieval)

Everything so far does **one fixed retrieval** per question. **Agentic RAG** puts
the LLM in a loop where it *decides*: should I retrieve at all? what should I
search for? is this enough, or should I search again? It can rewrite the query,
retrieve multiple times, use multiple sources, and stop when satisfied.

Named variants you will encounter:

- **Self-RAG** — the model reflects on whether it needs to retrieve and whether
  its draft is actually supported, re-retrieving if not. (Improves *reasoning*
  over evidence.)
- **Corrective RAG (CRAG)** — a lightweight grader scores retrieved docs and, if
  they are weak, discards them and falls back (e.g. to web search) before
  generating. (Improves the *evidence*.)
- **Multi-hop / decomposition** — break a question needing chained facts into
  sub-questions, retrieve per hop, synthesize.

**The catch: cost.** Agentic RAG can use **3–10×** the tokens and latency of
classic RAG. For a simple factual lookup that is pure waste.

**The 2026 consensus: adaptive / router RAG.** Classify each query by
difficulty. Route simple factual questions to cheap classic RAG; route hard,
multi-step questions to the agentic loop. In practice this often looks like an
**agent that has RAG as one tool** it calls *when it decides retrieval is
needed* — not a pipeline that always retrieves. Agentic RAG is a powerful
addition, not a replacement for the basics.

## 3. GraphRAG (answer "big picture" questions)

Vector RAG retrieves a few chunks — great for "what is the attachment limit?",
useless for "what are the main themes across all our incident reports?" No single
chunk contains a *global* answer.

**GraphRAG** (Microsoft) builds a **knowledge graph** at ingestion: an LLM
extracts entities and relationships from the text into nodes and edges, clusters
them into **communities**, and writes **summaries** of each community. At query
time:

- **Global search** (holistic questions) map-reduces over the community summaries.
- **Local search** (entity questions) walks the graph around the matched
  entities.

It shines on **aggregative / whole-corpus** questions that flat retrieval can't
do. The trade-off is a heavy, LLM-intensive indexing step. **LazyGraphRAG** is a
newer variant that defers most of that work to query time, slashing indexing cost
dramatically at similar quality — worth knowing if GraphRAG's index cost scares
you off.

Use GraphRAG only when your users genuinely ask corpus-spanning "themes /
patterns / summarize everything about X" questions. For lookup-style Q&A, plain
hybrid + rerank is simpler and cheaper.

## 4. When to adopt a framework

We built everything from scratch on purpose — you now know what each moving part
does. In production you may want a framework:

| Framework | What it's for |
| --- | --- |
| **LangChain / LangGraph** (v1.x) | Composable chains and, via LangGraph, stateful **agentic** control loops (branching, retries, human-in-the-loop). |
| **LlamaIndex** (v0.14.x) | Data/RAG-first: loaders, indexes, retrievers, query engines over your data. |
| **Haystack** (v2.x) | Explicit, typed, production-oriented pipelines. |
| **DSPy** (v3.x) | "Program, don't prompt" — declare the task and let optimizers tune the prompts/weights against a metric. A different axis; often used *with* the others. |

**When they help:** integrations (dozens of loaders/DBs for free), state and
observability, and not reinventing plumbing. **When they hurt:** early on, their
abstractions hide the very mechanics you need to understand to debug. The healthy
path is the one this course took — **build it raw once, then adopt a framework**
knowing exactly what it is doing for you. Frameworks change fast; the concepts
underneath (this whole course) do not.

## 5. Lab 8 — implement Contextual Retrieval and measure it

[`labs/lab08_contextual.py`](./labs/lab08_contextual.py) builds two indexes over
the corpus — **plain chunks** and **context-prepended chunks** — and compares
retrieval quality on the evaluation set from Lesson 7 (recall@k, MRR). It writes
each chunk's context blurb with your local LLM if one is available, and falls
back to a deterministic template (document title + section) otherwise, so it runs
and measures either way.

```bash
cd labs && source .venv/bin/activate
python lab08_contextual.py
```

## Key takeaways

- **Contextual retrieval** adds an LLM-written context blurb to each chunk before
  indexing — a cheap (with caching), ingestion-side fix for chunks that lost
  their meaning. It stacks with hybrid + rerank.
- **Agentic RAG** (self-RAG, CRAG, multi-hop) lets the model drive retrieval in a
  loop — powerful but 3–10× the cost. Use **adaptive routing**: classic RAG for
  easy queries, agentic for hard ones.
- **GraphRAG** answers whole-corpus "themes" questions via a knowledge graph and
  community summaries; **LazyGraphRAG** cuts its indexing cost. Overkill for
  lookup Q&A.
- Adopt a **framework** (LangChain/LlamaIndex/Haystack/DSPy) *after* you
  understand the raw pipeline — for integrations and state, not to learn.

## Further reading

- Contextual Retrieval — <https://www.anthropic.com/engineering/contextual-retrieval>
- GraphRAG — <https://microsoft.github.io/graphrag/> · LazyGraphRAG — <https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/>
- Self-RAG <https://arxiv.org/abs/2310.11511> · CRAG <https://arxiv.org/abs/2401.15884>

Next: **[Lesson 9 — Production RAG (capstone)](./09-production-rag.md)**.
