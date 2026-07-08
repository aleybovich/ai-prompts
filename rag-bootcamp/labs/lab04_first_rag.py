"""
Lab 4 — Your first end-to-end RAG pipeline.

Builds a persistent Chroma index once, then defines a reusable answer() function:
    retrieve top-k chunks -> build a grounded prompt -> call the LLM
and returns the ANSWER plus the SOURCES it used.

Run it:   python lab04_first_rag.py
No LLM needed to see retrieval + the assembled prompt (MockLLM prints it).
Install Ollama (`ollama pull gemma3:1b`) for real answers.
"""

import os

import chromadb

import ragkit

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
TOP_K = 4

embedder = ragkit.get_embedder()
llm = ragkit.get_llm()


def build_index():
    """
    Ingest ONCE: chunk + embed the corpus into a persistent Chroma collection.
    On later runs the collection already exists on disk, so we reuse it instead
    of re-embedding — this is the "ingest once, query many" idea in action.
    (Delete ./chroma_db, or the 'rag' collection, to force a rebuild.)
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    if any(c.name == "rag" for c in client.list_collections()):
        col = client.get_collection("rag")
        if col.count() > 0:
            print("Reusing the existing index from disk (ingest once, query many).")
            return col
        client.delete_collection("rag")
    col = client.create_collection("rag", metadata={"hnsw:space": "cosine"})
    chunks = ragkit.chunk_corpus()
    vecs = embedder.encode([c["text"] for c in chunks])
    col.add(
        ids=[c["id"] for c in chunks],
        embeddings=vecs.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"], "title": c["title"]} for c in chunks],
    )
    print("Built a fresh index on disk.")
    return col


SYSTEM = (
    "You are a support assistant for a product called Nimbus Notes. "
    "Answer the question using ONLY the context provided. "
    "If the context does not contain the answer, say you don't know. "
    "Cite the [source] you used for each fact."
)


def build_prompt(question, retrieved):
    context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def answer(question, col, k=TOP_K):
    """The whole RAG query path in one function (self-contained: pass in the store)."""
    qv = embedder.encode([question])[0].tolist()
    res = col.query(query_embeddings=[qv], n_results=k)
    retrieved = [
        {"source": m["source"], "text": d, "sim": 1 - dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    prompt = build_prompt(question, retrieved)
    text = llm.generate(prompt, system=SYSTEM)
    return {"answer": text, "sources": retrieved}


# --- Build the index, then answer several questions -----------------------
col = build_index()
print(f"Index ready: {col.count()} chunks.\n")

questions = [
    "Which plan includes SSO, and how much does it cost per user?",
    "How large can a single attachment be?",
    "What is Nimbus Notes' phone number?",   # <- deliberately NOT in the docs
]

for q in questions:
    result = answer(q, col)
    print("=" * 70)
    print("Q:", q)
    print("\nRetrieved sources (top-k):")
    for r in result["sources"]:
        print(f"  - ({r['sim']:+.2f}) {r['source']}: {' '.join(r['text'].split())[:60]}...")
    print("\nAnswer:")
    print(result["answer"])
    print()

print("=" * 70)
print("Notice the 3rd question: the answer is NOT in the docs. A good RAG system")
print("should say it doesn't know rather than invent a phone number. Whether it")
print("does depends on the model AND the prompt — that's Lesson 6.")
