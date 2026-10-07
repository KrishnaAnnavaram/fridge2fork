"""Ingredient embeddings and recipe vectors.

* ``SVDEmbedder`` (default, offline): positive PMI of ingredient co-occurrence in recipes,
  reduced with a seeded truncated SVD (scikit-learn).
* ``Word2VecEmbedder`` (extra ``w2v``): gensim Word2Vec with a fixed seed and one worker.

A recipe vector is the normalised mean of the normalised vectors of its known ingredients. If no
ingredient is known, the vector is ``None``. It is never a zero vector.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


def _unit(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class Embedder:
    name = "base"

    def __init__(self, dim: int = 64, seed: int = 42, min_count: int = 2):
        self.dim, self.seed, self.min_count = dim, seed, min_count
        self.vocab: dict[str, int] = {}
        self.vectors = np.zeros((0, dim), dtype=np.float32)

    def fit(self, recipes: list[list[str]]) -> "Embedder":
        raise NotImplementedError

    def _vocab(self, recipes: list[list[str]]) -> None:
        counts = Counter(i for r in recipes for i in set(r))
        names = sorted(n for n, c in counts.items() if c >= self.min_count)
        if len(names) < 2:
            raise ValueError("fewer than two ingredients reach min_count; lower FRIDGE2FORK_MIN_COUNT")
        self.vocab = {n: i for i, n in enumerate(names)}

    def vector(self, ingredients: list[str]) -> np.ndarray | None:
        idx = [self.vocab[i] for i in ingredients if i in self.vocab]
        if not idx:
            return None
        return _unit(self.vectors[idx].mean(axis=0)).astype(np.float32)

    def recipe_matrix(self, recipes: list[list[str]]) -> tuple[np.ndarray, np.ndarray]:
        """Vectors for all recipes and a mask of recipes that have at least one known ingredient."""
        out = np.zeros((len(recipes), self.vectors.shape[1]), dtype=np.float32)
        ok = np.zeros(len(recipes), dtype=bool)
        for r, ings in enumerate(recipes):
            v = self.vector(ings)
            if v is not None:
                out[r], ok[r] = v, True
        return out, ok


class SVDEmbedder(Embedder):
    name = "svd"

    def fit(self, recipes):
        self._vocab(recipes)
        n = len(self.vocab)
        rows, cols = [], []
        for r, ings in enumerate(recipes):
            ids = {self.vocab[i] for i in ings if i in self.vocab}
            rows += [r] * len(ids)
            cols += list(ids)
        X = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(recipes), n))
        co = (X.T @ X).tocoo()  # co-occurrence counts; the diagonal holds the single counts
        single = np.asarray(X.sum(axis=0)).ravel()
        off = co.row != co.col
        r_, c_, v_ = co.row[off], co.col[off], co.data[off]
        total = max(1.0, v_.sum())
        p_i = single / single.sum()
        pmi = np.log((v_ / total) / (p_i[r_] * p_i[c_]))
        keep = pmi > 0
        mat = sparse.csr_matrix((pmi[keep], (r_[keep], c_[keep])), shape=(n, n))
        k = max(1, min(self.dim, n - 1))
        svd = TruncatedSVD(n_components=k, random_state=self.seed)
        emb = svd.fit_transform(mat) if mat.nnz else np.eye(n, k)
        self.vectors = _unit(emb).astype(np.float32)
        return self


class Word2VecEmbedder(Embedder):
    name = "word2vec"

    def fit(self, recipes):
        try:
            from gensim.models import Word2Vec  # noqa: PLC0415  (optional extra)
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError('the word2vec embedder needs: pip install "fridge2fork[w2v]"') from exc
        self._vocab(recipes)
        sentences = [[i for i in r if i in self.vocab] for r in recipes]
        model = Word2Vec(sentences=sentences, vector_size=self.dim, window=10, min_count=1, sg=1,
                         seed=self.seed, workers=1, epochs=20)
        self.vectors = _unit(np.vstack([model.wv[n] for n in self.vocab])).astype(np.float32)
        return self


def make_embedder(name: str, dim: int, seed: int, min_count: int) -> Embedder:
    kinds = {"svd": SVDEmbedder, "word2vec": Word2VecEmbedder}
    if name not in kinds:
        raise ValueError(f"unknown embedder {name!r}")
    return kinds[name](dim=dim, seed=seed, min_count=min_count)
