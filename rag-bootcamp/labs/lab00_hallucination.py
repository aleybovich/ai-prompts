"""
Lab 0 — A complete RAG system in ~40 lines.

Answers a question about the sample "Nimbus Notes" docs two ways:
  1. WITHOUT RAG — ask the model directly (it has never heard of Nimbus Notes).
  2. WITH RAG    — retrieve the relevant passage, add it to the prompt, then ask.

Run it:   python lab00_hallucination.py
Runs with no LLM installed (prints the prompts). Install Ollama to see real
answers. Everything here is explained in detail in Lessons 1-6 — this is the
whole idea in one file.
"""

import ragkit

QUESTION = "How much does the Nimbus Notes Pro plan cost per month?"

# --- Ingestion: load docs, split into paragraph-sized chunks, embed them. ---
# (Proper chunking is Lesson 2; a paragraph split is fine to get started.)
docs = ragkit.load_corpus()
chunks = []
for d in docs:
    for para in d["text"].split("\n\n"):
        para = para.strip()
        if len(para) > 40:                     # skip tiny fragments/headers
            chunks.append({"source": d["id"], "text": para})

embedder = ragkit.get_embedder()
chunk_vecs = embedder.encode([c["text"] for c in chunks])

# --- Retrieval: embed the question, find the most similar chunks. ---
q_vec = embedder.encode([QUESTION])[0]
hits = ragkit.top_k(q_vec, chunk_vecs, k=3)
retrieved = [chunks[i] for i, _ in hits]

context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in retrieved)

llm = ragkit.get_llm()

print("=" * 70)
print("QUESTION:", QUESTION)
print("=" * 70)

# --- 1. WITHOUT RAG -------------------------------------------------------
print("\n--- WITHOUT RAG (model answers from memory) ---\n")
no_rag = llm.generate(QUESTION)
print(no_rag)

# --- 2. WITH RAG ----------------------------------------------------------
print("\n--- WITH RAG (model answers from retrieved context) ---\n")
system = (
    "You answer questions about Nimbus Notes using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know."
)
rag_prompt = f"Context:\n{context}\n\nQuestion: {QUESTION}\n\nAnswer:"
with_rag = llm.generate(rag_prompt, system=system)
print(with_rag)

print("\n" + "=" * 70)
print("Retrieved chunks came from:", [c["source"] for c in retrieved])
print("Notice: RAG did not change the model — it changed the PROMPT by")
print("inserting the right passage. That is the whole idea.")
