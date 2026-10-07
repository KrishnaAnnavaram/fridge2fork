"""Offline evaluation: hide ingredients of a recipe and check if the recipe comes back.

1. Select ``n_queries`` held-out recipes (seeded) with at least ``hide + 2`` ingredients.
2. Fit the embedder on the other recipes only. The index holds all recipes.
3. For each held-out recipe, hide ``hide`` random ingredients. The rest is the query.
4. Rank with each variant and record the rank of the held-out recipe.

Variants: ``dense`` (cosine only), ``overlap`` (coverage, use and missing, no cosine) and
``hybrid`` (the default weights).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from .config import Settings
from .rank import Weights
from .recommender import Recommender

VARIANTS = {
    "dense": Weights(coverage=0.0, use=0.0, dense=1.0, missing=0.0),
    "overlap": Weights(dense=0.0),
    "hybrid": Weights(),
}


def make_queries(recipes: pd.DataFrame, n_queries: int, hide: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = recipes[recipes["ingredients"].map(len) >= hide + 2]
    if pool.empty:
        raise ValueError("no recipe has enough ingredients for this --hide value")
    pick = pool.sample(n=min(n_queries, len(pool)), random_state=seed)
    rows = []
    for _, r in pick.iterrows():
        ings = list(r["ingredients"])
        hidden = set(rng.choice(ings, size=hide, replace=False).tolist())
        rows.append({"recipe_id": r["recipe_id"], "query": [i for i in ings if i not in hidden],
                     "hidden": sorted(hidden)})
    return pd.DataFrame(rows)


def evaluate(recipes: pd.DataFrame, settings: Settings, n_queries: int = 300, hide: int = 2,
             depth: int = 50) -> pd.DataFrame:
    """Recall@1/5/10 and MRR@depth per variant."""
    queries = make_queries(recipes, n_queries, hide, settings.seed)
    held = set(queries["recipe_id"])
    fit_on = recipes[~recipes["recipe_id"].isin(held)]
    base = Recommender.build(recipes, replace(settings, max_missing=None), fit_on=fit_on)
    rows = []
    for name, w in VARIANTS.items():
        base.weights = w
        ranks = []
        for _, q in queries.iterrows():
            res = base.recommend(q["query"], k=depth, max_missing=None)
            ids = [r.recipe_id for r in res.recommendations]
            ranks.append(ids.index(q["recipe_id"]) + 1 if q["recipe_id"] in ids else None)
        r = np.array([x if x is not None else np.inf for x in ranks], dtype=float)
        rows.append({
            "variant": name, "queries": len(r), "hide": hide,
            "recall@1": float(np.mean(r <= 1)), "recall@5": float(np.mean(r <= 5)),
            "recall@10": float(np.mean(r <= 10)), f"mrr@{depth}": float(np.mean(np.where(np.isfinite(r), 1 / r, 0))),
        })
    return pd.DataFrame(rows)
