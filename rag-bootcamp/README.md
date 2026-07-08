# The RAG Bootcamp

A hands-on, from-zero course on **Retrieval-Augmented Generation (RAG)** for
software engineers.

You do not need any prior machine-learning or AI background. If you have used a
coding assistant like Claude Code or Codex and can read Python, you are ready.
By the end you will understand how RAG works, be able to build a real RAG system,
and know how to make it accurate, cheap, and safe enough to use at work.

Everything in this course is **free**, runs **locally** on a normal laptop, and
uses **no paid subscriptions or licenses**. A local LLM (via Ollama) and small
open embedding models do all the work. Claude is shown once or twice as an
optional upgrade, never as a requirement.

> **Freshness.** This material was written in mid-2026 and names the current
> tools and models (sentence-transformers, Chroma 1.5, Cohere Rerank 4, Voyage 4
> embeddings, LangChain 1.x, Ragas 0.4, Gemma 3, Claude Opus 4.8, and so on). The
> *concepts*
> change slowly; specific version numbers and model names move fast, so treat
> them as a snapshot and check a tool's docs before you depend on a detail.

---

## What is RAG, in one paragraph

A large language model (LLM) only knows what it saw during training. It does not
know your company's documents, last week's news, or the contents of a specific
PDF — and when asked anyway, it often makes something up. **Retrieval-Augmented
Generation** fixes this by *retrieving* the relevant text from your own data and
*handing it to the model* as part of the prompt, so the model answers from real,
current, cite-able sources instead of memory. RAG is the standard way to build
question-answering, chatbots, and assistants over private or changing knowledge.

---

## Who this is for, and what you will be able to do

**For:** backend/full-stack/data engineers new to AI who want to build search and
question-answering over their own documents.

**After finishing, you will be able to:**

- Explain when RAG is the right tool — and when a bigger context window,
  fine-tuning, or plain caching is better.
- Turn documents into embeddings and run semantic search.
- Chunk documents well and store them in a vector database.
- Build a complete RAG pipeline end to end.
- Improve retrieval with hybrid search, reranking, and query rewriting.
- Write prompts that stay grounded in your data and cite their sources.
- Measure a RAG system with real metrics instead of vibes.
- Recognize and apply advanced patterns (contextual retrieval, agentic RAG,
  GraphRAG) and ship a system that is fast, cheap, and resistant to prompt
  injection.

---

## Glossary — read this first

Skim it now; every term is explained again where it is used. This is your
reference for the whole course.

**The building blocks**

- **LLM (large language model)** — a model like Claude, GPT, Llama, or Gemma that
  predicts text. It powers the "generation" half of RAG.
- **Token** — the unit an LLM reads and writes; roughly ¾ of a word. Costs and
  limits are measured in tokens.
- **Context window** — the maximum number of tokens a model can consider at once
  (today, up to ~1M for frontier models). Everything you want the model to use
  must fit here.
- **Prompt** — the text you send the model. In RAG, the prompt is assembled from
  a system instruction, the retrieved documents, and the user's question.
- **Knowledge cutoff** — the date after which the model has seen no training data.
  It cannot know anything newer without help.
- **Hallucination** — when a model states something false or unsupported with
  confidence. RAG reduces (but does not eliminate) hallucination.
- **Grounding** — making the model answer from provided source text rather than
  its own memory. A "grounded" answer can be traced back to a document.

**Retrieval side**

- **Corpus** — your whole collection of source documents.
- **Document** — one item in the corpus (a file, a page, an article).
- **Chunk** — a small slice of a document (a few sentences to a few paragraphs).
  RAG retrieves and stores chunks, not whole documents.
- **Chunking** — the process of splitting documents into chunks.
- **Embedding** — a list of numbers (a *vector*) that represents the *meaning* of
  a piece of text. Similar meanings produce nearby vectors.
- **Vector / dimension** — an embedding is a vector; its length (e.g. 384, 768,
  1536) is its number of *dimensions*.
- **Cosine similarity** — the standard way to measure how close two embeddings
  are (1.0 = identical direction, 0 = unrelated). This is how "semantic search"
  ranks results.
- **Semantic search** — finding text by *meaning* (via embeddings), so "how do I
  work without wifi?" can match a doc titled "offline mode."
- **Ingestion / indexing** — the offline step that loads, chunks, embeds, and
  stores your corpus so it can be searched later.
- **Vector database** — a store built to hold embeddings and find the nearest
  ones to a query quickly (e.g. Chroma, Qdrant, pgvector, LanceDB).
- **Index** — the data structure inside a vector database that makes search fast.
- **kNN (k-nearest-neighbors)** — returning the *k* closest vectors to a query.
- **ANN (approximate nearest neighbor)** — a faster, slightly-less-exact kNN used
  at scale. **HNSW** and **IVF** are common ANN index types.
- **top-k** — how many results you retrieve (e.g. "get the top 5 chunks").
- **BM25** — a classic keyword-search ranking (exact word matching). Great for
  codes, names, and error strings; blind to meaning/synonyms.
- **Dense vs. sparse** — dense retrieval = embeddings (meaning); sparse retrieval
  = keyword methods like BM25 (exact terms).
- **Hybrid search** — combining dense and sparse retrieval to get the best of
  both.
- **RRF (reciprocal rank fusion)** — a simple, robust way to merge two ranked
  lists (used to combine dense + sparse results).
- **Reranker / cross-encoder** — a second-stage model that re-scores a handful of
  candidate chunks by reading each one *together with* the query, for much better
  precision than embeddings alone.
- **Metadata filtering** — restricting search by structured fields (date, author,
  document type, permissions) alongside the vector search.
- **MMR (maximal marginal relevance)** — a way to pick results that are relevant
  *and* diverse, avoiding five near-duplicate chunks.
- **Query transformation** — rewriting or expanding the user's question before
  retrieval (e.g. **multi-query**, **HyDE**, **decomposition**) to retrieve
  better.

**Generation and quality**

- **Citation / attribution** — pointing each claim in the answer back to the
  chunk that supports it.
- **Faithfulness / groundedness** — whether the answer is actually supported by
  the retrieved text (vs. made up).
- **Context precision / recall** — did retrieval fetch the *right* chunks
  (precision) and *all* the needed chunks (recall)?
- **Evaluation** — measuring a RAG system's quality with metrics instead of
  guesswork.

**Advanced and production**

- **Contextual retrieval** — an ingestion trick (from Anthropic) that adds a
  short LLM-written context blurb to each chunk before embedding, sharply
  reducing retrieval failures.
- **Agentic RAG** — letting the LLM decide *whether and what* to retrieve, in a
  loop, instead of always doing one fixed retrieval.
- **GraphRAG** — building a knowledge graph from the corpus so the system can
  answer big "what are the themes across everything?" questions.
- **Prompt caching** — reusing the model's processing of a stable prefix (like a
  fixed set of documents) to cut cost and latency on repeated queries.
- **Fine-tuning** — further-training a model to change its style or skill. A
  different tool from RAG (RAG adds *knowledge*; fine-tuning changes *behavior*).
- **Prompt injection** — a security attack where malicious text *inside a
  retrieved document* tries to hijack the model's instructions.

---

## Course outline

Work through these in order. Each lesson is a Markdown file here; most come with
a runnable lab in [`labs/`](./labs/).

| # | Lesson | Lab | You will learn |
| --- | --- | --- | --- |
| 0 | [What is RAG and when to use it](./00-what-is-rag.md) | `lab00_hallucination.py` | The problem RAG solves; RAG vs. long context vs. fine-tuning |
| 1 | [Embeddings and semantic search](./01-embeddings-and-semantic-search.md) | `lab01_embeddings.py` | How text becomes vectors; cosine similarity; your first search |
| 2 | [Chunking and ingestion](./02-chunking-and-ingestion.md) | `lab02_chunking.py` | Splitting documents well; loading PDFs; overlap |
| 3 | [Vector databases and indexing](./03-vector-stores-and-indexing.md) | `lab03_vector_store.py` | Storing embeddings; ANN indexes; metadata filters (Chroma) |
| 4 | [Your first RAG pipeline](./04-first-rag-pipeline.md) | `lab04_first_rag.py` | Retrieve → augment → generate, end to end |
| 5 | [Better retrieval](./05-better-retrieval.md) | `lab05_hybrid_rerank.py` | Hybrid search + RRF, reranking, query rewriting |
| 6 | [Grounded generation and citations](./06-generation-and-citations.md) | `lab06_grounded_answers.py` | Prompts that cite sources and say "I don't know" |
| 7 | [Evaluating RAG](./07-evaluation.md) | `lab07_eval.py` | Retrieval and answer metrics; building an eval set |
| 8 | [Advanced patterns](./08-advanced-patterns.md) | `lab08_contextual.py` | Contextual retrieval, agentic RAG, GraphRAG, frameworks |
| 9 | [Production RAG (capstone)](./09-production-rag.md) | `lab09_capstone.py` | Cost, caching, latency, security, updates, monitoring |

## How to run the labs

See [`labs/README.md`](./labs/README.md) for setup (create a virtual
environment — never install as root), the free local model choices, and an
offline mode. The short version:

```bash
cd labs
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python lab00_hallucination.py     # then lab01_embeddings.py, lab02_..., through lab09
```

## A note on honesty

RAG is powerful but not magic. Throughout, the course flags where vendor claims
are marketing, where a technique helps only sometimes, and where RAG can still
get things wrong. Building the intuition for *when* each technique earns its
complexity is the real skill this bootcamp teaches.
