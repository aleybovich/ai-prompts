# Labs — setup and how to run them

Every lab is a single self-contained Python script. They are designed to be
**free**, **local**, and to run on a **normal laptop** — no paid accounts, no
GPU, no cloud services required.

## 1. Prerequisites

- Python 3.10 or newer (`python3 --version`).
- About 1 GB of free disk for the small models the labs download on first use.
- A terminal.

## 2. Create a virtual environment (never install as root)

Install everything into a project-local virtual environment. **Do not use
`sudo` and do not install packages system-wide.**

```bash
cd rag-bootcamp/labs
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When you are done, run `deactivate` to leave the virtual environment.

## 3. Run a lab

With the virtual environment active:

```bash
python lab01_embeddings.py
```

Each lab prints what it is doing and why. Read the output alongside the matching
lesson in the folder above.

## 4. The two pluggable pieces: embeddings and the LLM

Two parts of a RAG system depend on your machine and network. The labs keep them
behind a tiny helper module, `ragkit.py`, so the lab code stays focused on RAG
itself. You rarely touch `ragkit.py`, but it is short and worth reading.

### Embeddings (turning text into vectors)

By default the labs use **sentence-transformers** (the industry-standard local
embedding library, built on PyTorch) with the small `BAAI/bge-small-en-v1.5`
model (~130 MB, downloaded once, runs fine on CPU / Apple Silicon). No
configuration needed. If you are very tight on disk, use **fastembed** *instead
of* sentence-transformers (it is torch-free and much lighter): comment out the
`sentence-transformers` line in `requirements.txt`, run `pip install fastembed`,
and the labs will use it. (If both are installed, sentence-transformers takes
precedence.) To try a stronger model: `export RAGKIT_EMBED_MODEL=BAAI/bge-base-en-v1.5`.

### The LLM (generating answers)

Retrieval-only labs (Lessons 1–5) need **no LLM at all**. The labs that generate
answers (Lessons 6–9) will use, in order of preference:

1. **Ollama** — free, local. This is the recommended option. Install it from
   <https://ollama.com>, then pull a small model that fits a modest laptop:

   ```bash
   ollama pull gemma3:1b      # ~815 MB, a good small default
   # even smaller: ollama pull gemma3:270m   (~292 MB)
   # alternatives: ollama pull llama3.2:1b   /   ollama pull qwen2.5:1.5b
   ```

   Ollama runs a local server automatically; the labs talk to it. To pick a
   model: `export RAGKIT_OLLAMA_MODEL=gemma3:270m`.

2. **Claude (optional, paid)** — only if you set `ANTHROPIC_API_KEY`. This is an
   optional upgrade, never required. `pip install anthropic` first.

3. **MockLLM** — if neither is available, the labs still run and simply print the
   exact prompt they *would* send to a model. This lets you complete every lab
   with zero setup and see precisely what RAG assembles.

## 5. Offline / air-gapped mode

If you cannot download models (restricted network, CI, a quick smoke test), set:

```bash
export RAGKIT_OFFLINE=1
```

This swaps in a dependency-free **hashing** embedder and a lexical reranker so
every lab runs end to end with zero downloads.

> ⚠️ The offline embedder is a lexical trick (it matches shared *words*), **not**
> a real semantic model. Synonyms like "car" and "automobile" will *not* look
> similar. Use it only to check that code runs — install the real model
> (the default) to actually see semantic search working.

## 6. Environment variables (all optional)

| Variable | Effect |
| --- | --- |
| `RAGKIT_OFFLINE=1` | Use the offline hashing embedder / lexical reranker. |
| `RAGKIT_EMBED_MODEL=...` | Choose a different embedding model. |
| `RAGKIT_RERANK_MODEL=...` | Choose a different reranker model. |
| `RAGKIT_LLM=ollama\|anthropic\|mock` | Force a specific LLM backend. |
| `RAGKIT_OLLAMA_MODEL=gemma3:270m` | Pick the Ollama model. |
| `ANTHROPIC_API_KEY=...` | Enable the optional Claude backend. |

## 7. Files

- `ragkit.py` — shared helpers (embedder, reranker, LLM, corpus loader).
- `requirements.txt` — free, local dependencies.
- `data/corpus/*.md` — the sample knowledge base (the "Nimbus Notes" docs).
- `data/eval.json` — evaluation questions with ground-truth answers.
- `lab01_...` through `lab09_...` — one script per lesson.
