"""The recommender: build the indexes once, save them, load them, and answer queries.

Query procedure:

1. Map the user text to vocabulary names (normaliser, spelling correction, unknown list).
2. Candidates = recipes with the largest ingredient overlap UNION the dense top ``dense_k``.
3. Score each candidate with absolute features (``rank.py``), apply the filters, sort.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Settings
from .embed import Embedder, make_embedder
from .index import InvertedIndex, make_vector_index
from .normalize import QueryMatch, match_query
from .rank import Weights, allergens_of, is_vegetarian, score_recipe

FORMAT_VERSION = 1


@dataclass
class Recommendation:
    recipe_id: str
    name: str
    score: float
    coverage: float
    use: float
    dense: float | None
    matched: list[str]
    missing: list[str]
    allergens: list[str]
    minutes: float | None
    steps: list[str]
    ingredients_raw: list[str]


@dataclass
class Result:
    query: QueryMatch
    recommendations: list[Recommendation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    candidates: int = 0


class Recommender:
    def __init__(self, recipes: pd.DataFrame, embedder: Embedder, settings: Settings,
                 weights: Weights | None = None):
        self.recipes = recipes.reset_index(drop=True)
        self.embedder = embedder
        self.settings = settings
        self.weights = weights or Weights()
        lists = self.recipes["ingredients"].tolist()
        self._ids = self.recipes["recipe_id"].astype(str).tolist()
        self._names = self.recipes["name"].tolist()
        self._ings = lists
        self._steps = self.recipes["steps"].tolist()
        self._raw = self.recipes["ingredients_raw"].tolist()
        self._minutes = self.recipes["minutes"].tolist()
        self.vocabulary = sorted({i for r in lists for i in r})
        self.inverted = InvertedIndex(lists, self.vocabulary)
        self.vectors, self.valid = embedder.recipe_matrix(lists)
        self.vindex = make_vector_index(settings.vector_index, self.vectors, self.valid)

    # ------------------------------------------------------------ build / save / load
    @classmethod
    def build(cls, recipes: pd.DataFrame, settings: Settings, fit_on: pd.DataFrame | None = None,
              weights: Weights | None = None) -> "Recommender":
        """Fit the embedder (on ``fit_on`` if given, else on all recipes) and build the indexes."""
        emb = make_embedder(settings.embedder, settings.dim, settings.seed, settings.min_count)
        emb.fit((fit_on if fit_on is not None else recipes)["ingredients"].tolist())
        return cls(recipes, emb, settings, weights)

    def save(self, folder: str | Path, source: str | Path | None = None) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.recipes.to_pickle(folder / "recipes.pkl")
        np.save(folder / "ingredient_vectors.npy", self.embedder.vectors)
        (folder / "vocabulary.json").write_text(json.dumps(list(self.embedder.vocab)), encoding="utf-8")
        manifest = {
            "format_version": FORMAT_VERSION, "embedder": self.embedder.name, "dim": int(self.embedder.vectors.shape[1]),
            "seed": self.settings.seed, "min_count": self.settings.min_count, "recipes": len(self.recipes),
            "vocabulary": len(self.vocabulary), "embedded_ingredients": len(self.embedder.vocab),
            "recipes_without_vector": int((~self.valid).sum()),
            "source_sha256": _sha256(source) if source else None,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path, settings: Settings) -> "Recommender":
        folder = Path(folder)
        mpath = folder / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"no index in {folder}; run `fridge2fork build` first")
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError("the saved index has another format version; run `fridge2fork build` again")
        emb = make_embedder(manifest["embedder"], manifest["dim"], manifest["seed"], manifest["min_count"])
        names = json.loads((folder / "vocabulary.json").read_text(encoding="utf-8"))
        emb.vocab = {n: i for i, n in enumerate(names)}
        emb.vectors = np.load(folder / "ingredient_vectors.npy")
        with open(folder / "recipes.pkl", "rb") as fh:  # local file written by `save`
            recipes = pickle.load(fh)
        return cls(recipes, emb, settings)

    # ------------------------------------------------------------ query
    def recommend(self, text_or_list, k: int | None = None, max_missing: int | None = None,
                  exclude_allergens: tuple[str, ...] = (), vegetarian: bool = False,
                  exclude_ids: set[str] | None = None) -> Result:
        s = self.settings
        k = k or s.top_k
        max_missing = s.max_missing if max_missing is None else max_missing
        qm = match_query(text_or_list, set(self.vocabulary))
        res = Result(query=qm)
        if qm.corrections:
            res.warnings.append("corrected: " + ", ".join(f"{a} -> {b}" for a, b in qm.corrections.items()))
        if qm.unknown:
            res.warnings.append("not in the recipe data, ignored: " + ", ".join(qm.unknown))
        if not qm.known:
            res.warnings.append("no known ingredient in the query; no recommendation is possible")
            return res
        qvec = self.embedder.vector(qm.known)
        if qvec is None:
            res.warnings.append("no query ingredient has an embedding; only ingredient overlap is used")
        cand = set(self.inverted.top(qm.known, s.overlap_k).tolist())
        dense_score: dict[int, float] = {}
        if qvec is not None:
            ids, sims = self.vindex.search(qvec, s.dense_k)
            dense_score = dict(zip(ids.tolist(), sims.tolist()))
            cand |= set(dense_score)
        res.candidates = len(cand)
        recs = []
        for r in cand:
            if exclude_ids and self._ids[r] in exclude_ids:
                continue
            ings = self._ings[r]
            alg = allergens_of(ings)
            if set(alg) & set(exclude_allergens) or (vegetarian and not is_vegetarian(ings)):
                continue
            dense = dense_score.get(r)
            if dense is None and qvec is not None and self.valid[r]:
                dense = float(self.vectors[r] @ qvec)
            f = score_recipe(ings, qm.known, dense, self.weights, s.assume_pantry)
            if max_missing is not None and len(f.missing) > max_missing:
                continue
            minutes = self._minutes[r]
            recs.append(Recommendation(
                recipe_id=self._ids[r], name=self._names[r], score=round(f.score, 4),
                coverage=round(f.coverage, 4), use=round(f.use, 4),
                dense=None if f.dense is None else round(float(f.dense), 4), matched=f.matched, missing=f.missing,
                allergens=alg, minutes=None if pd.isna(minutes) else float(minutes), steps=list(self._steps[r]),
                ingredients_raw=list(self._raw[r])))
        recs.sort(key=lambda x: (-x.score, len(x.missing), x.recipe_id))
        res.recommendations = recs[:k]
        if not res.recommendations:
            res.warnings.append("no recipe passes the filters")
        return res


def result_to_dict(res: Result) -> dict:
    return {"query": asdict(res.query), "warnings": res.warnings, "candidates": res.candidates,
            "recommendations": [asdict(r) for r in res.recommendations]}


def _sha256(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
