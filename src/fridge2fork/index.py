"""Search structures: a dense vector index (cosine) and an inverted ingredient index."""
from __future__ import annotations

from typing import Protocol

import numpy as np
from scipy import sparse


class VectorIndex(Protocol):
    name: str

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...


class NumpyIndex:
    """Exact cosine search with one matrix product. The vectors must have length 1 (or be all zero)."""

    name = "numpy"

    def __init__(self, vectors: np.ndarray, valid: np.ndarray):
        self.vectors = np.asarray(vectors, dtype=np.float32)
        self.valid = np.asarray(valid, dtype=bool)

    def search(self, query, k):
        sims = self.vectors @ np.asarray(query, dtype=np.float32)
        sims[~self.valid] = -np.inf  # a recipe with no known ingredient is never a dense hit
        k = min(k, int(self.valid.sum()))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return top, sims[top]


class FaissIndex:
    """Exact inner-product search with FAISS (extra ``faiss``). Same results as ``NumpyIndex``."""

    name = "faiss"

    def __init__(self, vectors: np.ndarray, valid: np.ndarray):
        try:
            import faiss  # noqa: PLC0415  (optional extra)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('the faiss index needs: pip install "fridge2fork[faiss]"') from exc
        self.ids = np.flatnonzero(valid)
        v = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32)[self.ids])
        self.index = faiss.IndexFlatIP(v.shape[1])
        self.index.add(v)

    def search(self, query, k):
        k = min(k, len(self.ids))
        if k <= 0:
            return np.array([], dtype=int), np.array([], dtype=np.float32)
        sims, idx = self.index.search(np.asarray(query, dtype=np.float32)[None, :], k)
        return self.ids[idx[0]], sims[0]


def make_vector_index(name: str, vectors: np.ndarray, valid: np.ndarray) -> VectorIndex:
    if name == "numpy":
        return NumpyIndex(vectors, valid)
    if name == "faiss":
        return FaissIndex(vectors, valid)
    raise ValueError(f"unknown vector index {name!r}")


class InvertedIndex:
    """Recipe x ingredient incidence matrix. Gives the overlap count of a query with every recipe."""

    def __init__(self, recipes: list[list[str]], vocabulary: list[str]):
        self.vocab = {n: i for i, n in enumerate(vocabulary)}
        rows, cols = [], []
        for r, ings in enumerate(recipes):
            ids = {self.vocab[i] for i in ings if i in self.vocab}
            rows += [r] * len(ids)
            cols += list(ids)
        self.matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                                        shape=(len(recipes), len(vocabulary)))

    def overlap(self, ingredients: list[str]) -> np.ndarray:
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for i in ingredients:
            if i in self.vocab:
                q[self.vocab[i]] = 1
        return np.asarray(self.matrix @ q).ravel()

    def top(self, ingredients: list[str], k: int) -> np.ndarray:
        ov = self.overlap(ingredients)
        hits = np.flatnonzero(ov > 0)
        if len(hits) > k:
            hits = hits[np.argsort(-ov[hits], kind="stable")[:k]]
        return hits
