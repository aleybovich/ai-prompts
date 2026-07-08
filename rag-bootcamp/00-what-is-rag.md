# Lesson 0 — What is RAG, and when should you use it?

**Goal:** understand the problem RAG solves, see a complete RAG system in ~40
lines, and learn when RAG is the right tool versus the alternatives.

---

## 1. The problem: an LLM only knows what it was trained on

A large language model (LLM) is trained once on a huge pile of text, then frozen.
That gives it three problems the moment you try to use it on real work:

1. **Knowledge cutoff.** It has never seen anything published after its training
   date. Ask about last week's release and it cannot know.
2. **No private knowledge.** It has never seen your company wiki, your product
   docs, your codebase, or that PDF on your desktop.
3. **Hallucination.** When it does not know, it usually does not say "I don't
   know." It produces a fluent, confident, *wrong* answer. This is the single
   biggest reason naive LLM apps fail in production.

You could try to fix this by pasting all your documents into every prompt. But
your documents may be gigabytes; the model's **context window** (how much text it
can read at once) is large today but still finite, and stuffing it full is slow
and expensive. You need to send only the *relevant* pieces.

## 2. The idea: retrieve first, then generate

**Retrieval-Augmented Generation** is exactly that two-step idea:

1. **Retrieve** — given the user's question, search your own documents and pull
   out the handful of passages most likely to contain the answer.
2. **Generate** — put those passages into the prompt and ask the LLM to answer
   *using only that provided text*.

The model is no longer answering from memory. It is answering from source
material you handed it — material that can be private, up to date, and
**cite-able**. That is the whole trick, and everything else in this course is
about doing each of those two steps well.

```
                    ┌──────────────── INGESTION (done once, offline) ─────────────────┐
                    │  documents → chunk → embed → store in a vector database         │
                    └────────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
  user question ──► RETRIEVAL ──► top-k relevant chunks ──► build prompt ──► LLM ──► grounded answer
                    (search the vector database)             (chunks + question)
```

*(Don't worry about the new words yet: **chunk** = split a document into
passages; **embed** = turn text into a vector of numbers that captures its
meaning; **vector database** = a store you can search by meaning; **top-k** = the
few best matches. Each gets its own lesson next.)*

The top row (**ingestion**) happens ahead of time: you prepare your knowledge
once. The bottom row (**retrieval + generation**) happens on every question. Most
of the engineering effort — and most of this course — is in making retrieval find
the *right* chunks, because the generator can only be as good as what you feed it.

## 3. Lab 0 — a complete RAG system you can run now

Before we break RAG into pieces, run the whole thing once. The lab
[`labs/lab00_hallucination.py`](./labs/lab00_hallucination.py) answers a question
about our sample "Nimbus Notes" docs two ways:

- **Without RAG:** it asks the model the question directly. The model has never
  heard of Nimbus Notes, so it has no source for the answer — it guesses, makes
  something up, or admits it doesn't know. None of those are usable or cite-able.
- **With RAG:** it retrieves the most relevant chunks (the top few) from the
  docs, adds them to the prompt, and asks again. Now the answer is correct and
  traceable. (Retrieval is fuzzy — later lessons improve *which* chunks come
  back.)

```bash
cd labs && source .venv/bin/activate      # first time? see labs/README.md to set this up
python lab00_hallucination.py
```

Even with **no LLM installed**, the lab runs: it prints the exact prompt each
approach would send, so you can *see* that RAG's prompt contains the real answer
while the plain prompt does not. Install Ollama (see `labs/README.md`) to watch
a real model get it wrong, then right.

Key thing to notice: RAG did not change the *model*. It changed the *prompt* —
by finding and inserting the right context. RAG is a retrieval problem wearing a
generation hat.

## 4. When should you use RAG? (And when not?)

RAG is not the only way to give a model knowledge. Pick the right tool:

| Situation | Best tool | Why |
| --- | --- | --- |
| Large or frequently-changing private knowledge; you need citations | **RAG** | Only sends relevant, current, source-linked passages |
| Small, stable set of docs (rule of thumb: under ~50k words) | **Long context + prompt caching** | Just put it all in the prompt; cache it so you don't re-pay each call |
| You need to change the model's *style, format, or skill* (not its facts) | **Fine-tuning** | Teaches behavior, not knowledge |
| The same big static context is queried over and over | **Prompt caching** | Reuse the model's processing of the fixed part |

Two myths worth killing early:

- **"Context windows are huge now, so RAG is dead."** Not so. Sending 1M tokens
  every request is slow and expensive, and models get *worse* at using
  information buried in the middle of a very long context (the well-documented
  "lost in the middle" effect). Retrieval keeps the prompt short and on-target.
  The 2026 consensus is **not RAG vs. long context — it's both**: retrieve to
  narrow down, then let a capable model reason over the narrowed set.
- **"RAG eliminates hallucination."** It reduces it a lot, but models can still
  add unsupported detail or misread the context. Lessons 6 and 7 are about
  measuring and containing that.

## 5. What RAG is *not*

- It is **not fine-tuning.** You are not training anything. RAG works with an
  off-the-shelf model by changing its input.
- It is **not just keyword search.** Classic search matches words; RAG matches
  *meaning* (Lesson 1) and then feeds results to a model that writes an answer.
- It is **not a single library.** RAG is a pattern. You can build it from scratch
  (we will) or with a framework like LangChain or LlamaIndex (Lesson 8).

## Key takeaways

- An LLM only knows its training data; RAG supplies the rest by **retrieving**
  relevant text and putting it in the prompt.
- RAG = **ingestion** (chunk, embed, store — done once) + **retrieval +
  generation** (done per question).
- Retrieval quality is the ceiling on answer quality.
- RAG is one option among long-context, caching, and fine-tuning — choose based
  on data size, change rate, and whether you need citations.
- RAG reduces hallucination but does not remove it; you must measure quality.

## Further reading

- Anthropic, *Contextual Retrieval* — <https://www.anthropic.com/engineering/contextual-retrieval>
- "Lost in the Middle" (Liu et al., 2023) — how long contexts degrade — <https://arxiv.org/abs/2307.03172>

Next: **[Lesson 1 — Embeddings and semantic search](./01-embeddings-and-semantic-search.md)**, where we make the "match by meaning" step concrete.
