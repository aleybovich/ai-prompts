"""
ragkit.py — tiny shared helpers for the RAG Bootcamp labs.

Design goals
------------
1. FREE + LOCAL: no paid API, no subscription, runs on a normal Mac laptop.
2. PLUGGABLE: the two pieces that depend on your machine/network — the
   embedding model and the LLM — live here behind a small interface, so the
   lab code stays focused on RAG logic (chunking, retrieval, ranking, prompts).
3. ALWAYS RUNS: if the real embedding model or a local LLM isn't available,
   ragkit falls back to an OFFLINE backend so the code still runs end to end.
   The offline embedder is NOT semantically smart (see class docstring) — it's
   there so the pipeline never crashes and can be smoke-tested. Install the real
   model (instructions in labs/README.md) to see real semantic search.

You normally only call four things from labs:
    get_embedder()        -> an object with .encode(list[str]) -> np.ndarray
    get_llm()             -> an object with .generate(prompt, system=None) -> str
    cosine_sim(a, b)      -> float
    top_k(query_vec, matrix, k) -> list[(index, score)]
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Embedders
# ---------------------------------------------------------------------------

# Default embedding model: BAAI/bge-small-en-v1.5 (~130 MB, 384 dims). A modern,
# widely used sentence-transformers model that is clearly stronger than the older
# all-MiniLM-L6-v2 (~90 MB) while staying small and fast on a laptop. The name
# works with both sentence-transformers and fastembed. Bigger/better upgrades
# (still free/local): "BAAI/bge-base-en-v1.5" (~440 MB, 768 dims),
# "nomic-ai/nomic-embed-text-v1.5" (~550 MB, 8k context), "Qwen/Qwen3-Embedding-0.6B".
DEFAULT_EMBED_MODEL = os.environ.get("RAGKIT_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# Optional reranker-model override. Leave unset to let each backend use its own
# default (sentence-transformers: "cross-encoder/ms-marco-MiniLM-L-6-v2";
# fastembed: "Xenova/ms-marco-MiniLM-L-6-v2" — same model, ~80 MB).
RERANK_MODEL_OVERRIDE = os.environ.get("RAGKIT_RERANK_MODEL")


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so dot product == cosine similarity."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class FastEmbedEmbedder:
    """
    Real semantic embeddings via `fastembed` (Qdrant). Preferred default because
    it uses ONNX (no PyTorch), so it installs small and runs fast on a laptop.
    """

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from fastembed import TextEmbedding  # lazy import

        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        # embed one probe to learn the dimension
        probe = next(iter(self._model.embed(["probe"])))
        self.dim = int(np.asarray(probe).shape[-1])

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vecs = list(self._model.embed(list(texts)))
        return _l2_normalize(np.asarray(vecs, dtype=np.float32))


class SentenceTransformerEmbedder:
    """Real semantic embeddings via the `sentence-transformers` library (uses PyTorch)."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


class HashingEmbedder:
    """
    OFFLINE fallback embedder — a "bag of hashed words" vector.

    IMPORTANT: this is a LEXICAL trick, not a semantic model. It maps each word
    to a bucket by hashing, so texts that share the *same words* look similar,
    but true synonyms ("car" vs "automobile") do NOT. It exists so the labs run
    with zero downloads and can be smoke-tested. For real semantic search,
    install sentence-transformers (see labs/README.md).
    """

    # Common English words carry little meaning and, worse, cause spurious
    # "similarity" between unrelated sentences that merely share them. Dropping
    # them makes the offline stand-in less misleading (it is still not semantic).
    STOPWORDS = frozenset(
        "a an the of to in on at is are am be was were do does did i you it we they "
        "and or but for with without no not this that these those my your our their "
        "what how when where why who which as by from into out up down over under "
        "can could should would will may might have has had".split()
    )

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_name = f"offline-hashing-{dim}d"

    def _tokens(self, text: str) -> list[str]:
        return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in self.STOPWORDS]

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        rows = []
        for text in texts:
            v = np.zeros(self.dim, dtype=np.float32)
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
                # a few char-trigrams add a little fuzziness (small weight)
                for i in range(len(tok) - 2):
                    tri = tok[i : i + 3]
                    hh = int(hashlib.md5(tri.encode()).hexdigest(), 16)
                    v[hh % self.dim] += 0.15
            rows.append(v)
        return _l2_normalize(np.vstack(rows)) if rows else np.zeros((0, self.dim), np.float32)


_EMBEDDER = None


def get_embedder(model_name: str | None = None, verbose: bool = True):
    """
    Return a cached embedder. Uses the real sentence-transformers model unless
    RAGKIT_OFFLINE=1 is set or the library/model is unavailable, in which case
    it falls back to the offline HashingEmbedder.
    """
    global _EMBEDDER
    if _EMBEDDER is not None and model_name is None:
        return _EMBEDDER

    name = model_name or DEFAULT_EMBED_MODEL
    emb = None
    if os.environ.get("RAGKIT_OFFLINE") == "1":
        emb = HashingEmbedder()
        if verbose:
            print(f"[ragkit] embedder: {emb.model_name} (RAGKIT_OFFLINE=1)", file=sys.stderr)
    else:
        # Prefer sentence-transformers (the industry standard). Fall back to
        # fastembed (a lighter, torch-free alternative), then to the offline
        # hashing embedder so labs always run even with no models available.
        for backend in (SentenceTransformerEmbedder, FastEmbedEmbedder):
            try:
                emb = backend(name)
                if verbose:
                    print(
                        f"[ragkit] embedder: {name} via {backend.__name__} (dim={emb.dim})",
                        file=sys.stderr,
                    )
                break
            except Exception:
                continue
        if emb is None:
            emb = HashingEmbedder()
            if verbose:
                print(
                    "[ragkit] no embedding model available (library missing or "
                    f"download blocked); using {emb.model_name}. Install "
                    "sentence-transformers for real semantic search — see labs/README.md.",
                    file=sys.stderr,
                )
    if model_name is None:
        _EMBEDDER = emb
    return emb


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
    """
    Brute-force nearest neighbours. Returns [(row_index, score), ...] sorted by
    descending score. Assumes rows of `matrix` and `query_vec` are normalized
    (so a dot product is cosine similarity).
    """
    query_vec = np.asarray(query_vec, dtype=np.float32).ravel()
    scores = matrix @ query_vec
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return [(int(i), float(scores[i])) for i in idx]


# ---------------------------------------------------------------------------
# LLM backends (for the generation half of RAG)
# ---------------------------------------------------------------------------

class MockLLM:
    """
    No real model. Echoes the prompt so retrieval-only labs run with zero setup.
    Great for inspecting exactly what you send to a model.
    """

    name = "mock"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return (
            "[MockLLM] No local LLM found (install Ollama for real answers — see "
            "labs/README.md). Here is the prompt that WOULD be sent to the model:\n\n"
            + (f"SYSTEM:\n{system}\n\n" if system else "")
            + f"PROMPT:\n{prompt}"
        )


class OllamaLLM:
    """
    Free, local generation via Ollama (https://ollama.com). Run once:
        ollama pull gemma3:1b      # or gemma3:270m, llama3.2:1b, qwen2.5:1.5b
    Then Ollama serves a simple local HTTP API on http://localhost:11434.
    """

    def __init__(self, model: str | None = None, host: str | None = None):
        # gemma3:1b is ~815 MB and runs on a modest laptop. Even smaller:
        # gemma3:270m (~292 MB). Alternatives: llama3.2:1b, qwen2.5:1.5b.
        self.model = model or os.environ.get("RAGKIT_OLLAMA_MODEL", "gemma3:1b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.name = f"ollama:{self.model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        import json
        import urllib.request

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
        }
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"].strip()


class AnthropicLLM:
    """
    OPTIONAL upgrade — NOT required for this bootcamp. Uses Claude via the
    Anthropic API, which needs a paid API key in ANTHROPIC_API_KEY.
    """

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic  # lazy import

        self.client = anthropic.Anthropic()
        self.model = model
        self.name = f"anthropic:{model}"

    def generate(self, prompt: str, system: str | None = None) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system or "",
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()


_LLM = None


def get_llm(verbose: bool = True):
    """
    Return a cached LLM backend, preferring free/local options:
      1. Ollama, if a server is reachable (free, local — recommended).
      2. Anthropic/Claude, only if ANTHROPIC_API_KEY is set (optional, paid).
      3. MockLLM, which just echoes the prompt (zero setup).
    Force one with RAGKIT_LLM = ollama | anthropic | mock.
    """
    global _LLM
    if _LLM is not None:
        return _LLM

    choice = os.environ.get("RAGKIT_LLM", "").lower()

    def announce(llm):
        if verbose:
            print(f"[ragkit] llm: {llm.name}", file=sys.stderr)
        return llm

    if choice == "mock":
        _LLM = announce(MockLLM())
    elif choice == "anthropic":
        _LLM = announce(AnthropicLLM())
    elif choice == "ollama":
        _LLM = announce(OllamaLLM())
    else:
        # auto-detect: try Ollama, then Anthropic key, else mock
        if _ollama_up():
            _LLM = announce(OllamaLLM())
        elif os.environ.get("ANTHROPIC_API_KEY"):
            try:
                _LLM = announce(AnthropicLLM())
            except Exception:
                _LLM = announce(MockLLM())
        else:
            _LLM = announce(MockLLM())
    return _LLM


def _ollama_up() -> bool:
    import urllib.request

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    try:
        urllib.request.urlopen(host + "/api/tags", timeout=1.5)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reranker (used in the "better retrieval" lab)
# ---------------------------------------------------------------------------

class SentenceTransformerReranker:
    """Cross-encoder reranker via sentence-transformers (the industry standard)."""

    def __init__(self, model_name: str | None = None):
        from sentence_transformers import CrossEncoder  # lazy import

        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self._model = CrossEncoder(self.model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        pairs = [[query, d] for d in documents]
        return [float(s) for s in self._model.predict(pairs)]


class FastEmbedReranker:
    """Cross-encoder reranker via fastembed (ONNX, no torch) — lighter alternative."""

    def __init__(self, model_name: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy import

        self.model_name = model_name or "Xenova/ms-marco-MiniLM-L-6-v2"
        self._model = TextCrossEncoder(model_name=self.model_name)

    def scores(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


class LexicalReranker:
    """
    OFFLINE fallback reranker — scores by word overlap between query and doc.
    Not a real cross-encoder, but lets the reranking lab run with no download.
    """

    model_name = "offline-lexical-overlap"

    def scores(self, query: str, documents: list[str]) -> list[float]:
        q = set(re.findall(r"[a-z0-9]+", query.lower()))
        out = []
        for d in documents:
            words = re.findall(r"[a-z0-9]+", d.lower())
            if not words:
                out.append(0.0)
                continue
            hits = sum(1 for w in words if w in q)
            out.append(hits / (len(words) ** 0.5))
        return out


def get_reranker(model_name: str | None = None, verbose: bool = True):
    """
    Return a cross-encoder reranker. Prefers sentence-transformers, falls back to
    fastembed, then to an offline lexical reranker so the lab always runs.
    """
    name = model_name or RERANK_MODEL_OVERRIDE  # may be None -> backend default
    rr = None
    if os.environ.get("RAGKIT_OFFLINE") == "1":
        rr = LexicalReranker()
    else:
        for backend in (SentenceTransformerReranker, FastEmbedReranker):
            try:
                rr = backend(name)
                break
            except Exception:
                continue
        if rr is None:
            rr = LexicalReranker()
    if verbose:
        print(f"[ragkit] reranker: {rr.model_name}", file=sys.stderr)
    return rr


# ---------------------------------------------------------------------------
# Chunking (Lesson 2 builds these by hand; later labs reuse this helper)
# ---------------------------------------------------------------------------

def chunk_document(text: str, size: int = 1200, overlap: int = 180) -> list[str]:
    """
    Recursive-character chunking: split on the largest natural boundary that
    keeps pieces under `size` (paragraphs -> lines -> sentences -> words), then
    add `overlap` characters from the previous chunk. Sizes are in characters
    (~4 chars per token). Uses the same recursive algorithm as lab02's
    recursive_chunks.

    The default size (1200 chars ~= 300 tokens) is intentionally SMALLER than
    Lesson 2's general ~1600-2000 char (~400-512 token) recommendation, because
    the sample-corpus documents are short and we want each to split into a few
    chunks for the demos. On real documents, prefer the larger size.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def split(t: str, seps: list[str]) -> list[str]:
        if len(t) <= size or not seps:
            return [t]
        sep, rest = seps[0], seps[1:]
        parts = t.split(sep)
        out, buf = [], []
        buflen = 0
        for part in parts:
            add = len(part) + (len(sep) if buf else 0)
            if buf and buflen + add > size:
                out.append(sep.join(buf))       # rejoin with sep, no trailing sep
                buf, buflen = [], 0
            if len(part) > size:                # a single part too big: go finer
                if buf:
                    out.append(sep.join(buf))
                    buf, buflen = [], 0
                out.extend(split(part, rest))
            else:
                buf.append(part)
                buflen += add
        if buf:
            out.append(sep.join(buf))
        return out

    raw = [c.strip() for c in split(text, separators) if c.strip()]
    chunks = []
    for i, c in enumerate(raw):
        if i > 0 and overlap:
            c = raw[i - 1][-overlap:] + " " + c
        chunks.append(c)
    return chunks


def chunk_corpus(docs: list[dict] | None = None, size: int = 1200, overlap: int = 180) -> list[dict]:
    """
    Chunk every document in the corpus. Returns a list of chunk dicts:
    {id, source, title, pos, text} — carrying metadata for citations/filtering.
    """
    docs = docs if docs is not None else load_corpus()
    out = []
    for d in docs:
        pieces = chunk_document(d["text"], size=size, overlap=overlap)
        for pos, text in enumerate(pieces):
            out.append(
                {
                    "id": f"{d['id']}#{pos}",
                    "source": d["id"],
                    "title": d["title"],
                    "pos": pos,
                    "text": text,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Tiny corpus loader used by several labs
# ---------------------------------------------------------------------------

def load_corpus(path: str | None = None) -> list[dict]:
    """
    Load the sample corpus (one dict per document: {id, title, text}).
    Defaults to labs/data/corpus/*.md relative to this file.
    """
    import glob

    if path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "data", "corpus")
    docs = []
    for fp in sorted(glob.glob(os.path.join(path, "*.md"))):
        with open(fp, encoding="utf-8") as fh:
            text = fh.read()
        title = os.path.splitext(os.path.basename(fp))[0]
        first = text.strip().splitlines()[0] if text.strip() else title
        docs.append(
            {
                "id": os.path.basename(fp),
                "title": first.lstrip("# ").strip() or title,
                "text": text,
            }
        )
    return docs


if __name__ == "__main__":
    # Quick self-test: `python ragkit.py`
    emb = get_embedder()
    m = emb.encode(["the cat sat on the mat", "a kitten rests on a rug", "quantum field theory"])
    print("embedding matrix shape:", m.shape)
    print("cat vs kitten :", round(cosine_sim(m[0], m[1]), 3))
    print("cat vs quantum:", round(cosine_sim(m[0], m[2]), 3))
    if "offline" in emb.model_name:
        print("(offline lexical stand-in — numbers near 0 are expected; install "
              "sentence-transformers for meaningful semantic similarity)")
    print("llm backend   :", get_llm().name)
