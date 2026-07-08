"""
Lab 3 — A real vector database with Chroma.

  1. Chunk + embed the corpus.
  2. Store chunks, embeddings, and METADATA in a PERSISTENT Chroma collection.
  3. Query by meaning; then query again with a METADATA FILTER.
  4. Upsert (update a chunk) and re-query.
  5. Confirm Chroma returns the same top hit as brute-force search.

Run it:   python lab03_vector_store.py
The store is saved under ./chroma_db and reloads on the next run.
"""

import os

import chromadb

import ragkit

DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# --- Category metadata so we have something meaningful to filter on --------
CATEGORY = {
    "pricing.md": "billing",
    "api.md": "developer",
    "troubleshooting.md": "support",
    "security.md": "security",
    "sync-and-offline.md": "support",
    "sharing-and-collaboration.md": "collaboration",
    "import-export.md": "data",
    "getting-started.md": "onboarding",
    "changelog.md": "release",
    "overview.md": "general",
}

# --- 1. Chunk + embed ------------------------------------------------------
chunks = ragkit.chunk_corpus()
embedder = ragkit.get_embedder()
vecs = embedder.encode([c["text"] for c in chunks])
print(f"Prepared {len(chunks)} chunks, embedding dim {vecs.shape[1]}\n")

# --- 2. Store in a persistent Chroma collection ----------------------------
client = chromadb.PersistentClient(path=DB_PATH)
# Start clean each run so the demo is reproducible.
if any(c.name == "nimbus" for c in client.list_collections()):
    client.delete_collection("nimbus")
# cosine space matches our normalized embeddings (Lesson 3)
collection = client.create_collection("nimbus", metadata={"hnsw:space": "cosine"})

collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=vecs.tolist(),
    documents=[c["text"] for c in chunks],
    metadatas=[
        {"source": c["source"], "title": c["title"], "category": CATEGORY.get(c["source"], "general")}
        for c in chunks
    ],
)
print(f"Stored {collection.count()} chunks in Chroma at {DB_PATH}\n")


def show(results, header):
    print(header)
    ids = results["ids"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]
    docs = results["documents"][0]
    for rank, (cid, dist, meta, doc) in enumerate(zip(ids, dists, metas, docs), 1):
        sim = 1 - dist  # cosine distance -> similarity
        snippet = " ".join(doc.split())[:70]
        print(f"  {rank}. sim={sim:+.2f} [{meta['category']}] {cid}")
        print(f"     {snippet}...")
    print()


# --- 3. Semantic query, then the same query WITH a metadata filter ---------
query = "how is my data protected"
qv = embedder.encode([query])[0].tolist()

show(
    collection.query(query_embeddings=[qv], n_results=3),
    f"Query {query!r} (no filter):",
)
show(
    collection.query(query_embeddings=[qv], n_results=3, where={"category": "billing"}),
    f"Same query, but FILTERED to category='billing' "
    f"(note how the results change — filtering happens with the search):",
)

# --- 4. Chroma vs brute force: same math, faster path ----------------------
# (Run this BEFORE the upsert, while Chroma and our local `vecs` hold identical
# data, so "same math" is actually true rather than true by luck.)
bf = ragkit.top_k(embedder.encode([query])[0], vecs, k=1)[0]
bf_id = chunks[bf[0]]["id"]
ch_id = collection.query(query_embeddings=[qv], n_results=1)["ids"][0][0]
print("-" * 60)
print("Sanity check: brute-force top hit vs Chroma top hit")
print(f"  brute force: {bf_id}")
print(f"  chroma     : {ch_id}")
print(f"  match: {bf_id == ch_id}  (a vector DB is a faster route to the same result)\n")

# --- 5. Upsert: update an existing chunk, then re-query --------------------
# Pick the first pricing chunk so the update is coherent with its document.
pricing_chunk = next(c for c in chunks if c["source"] == "pricing.md")
new_text = "UPDATED: Nimbus now offers a 14-day free trial of the Pro plan."
collection.upsert(
    ids=[pricing_chunk["id"]],
    embeddings=[embedder.encode([new_text])[0].tolist()],
    documents=[new_text],
    metadatas=[{"source": "pricing.md", "title": pricing_chunk["title"], "category": "billing"}],
)
show(
    collection.query(query_embeddings=[embedder.encode(["free trial"])[0].tolist()], n_results=2),
    "After upsert, searching 'free trial' surfaces the updated chunk:",
)

# --- 6. Persistence: the data is really on disk ----------------------------
# Open a brand-new client pointing at the same path (as if the program restarted)
# and read the collection back without re-adding anything.
print("-" * 60)
reopened = chromadb.PersistentClient(path=DB_PATH)
reloaded = reopened.get_collection("nimbus")
print(f"Reopened the store from disk: {reloaded.count()} chunks are still there.")
print("(This lab wipes + rebuilds the collection at the START of each run purely")
print(" so the demo is reproducible. In a real app you build the index once and")
print(" reuse it across runs — the vectors persist under ./chroma_db.)")
