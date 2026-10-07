"""Load the recipe file and validate it.

The expected layout is the Food.com ``RAW_recipes.csv`` layout: ``name``, ``ingredients`` (a list
string) and, optionally, ``id``, ``steps``, ``minutes``, ``tags`` and ``description``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .normalize import canonical_list, parse_list

REQUIRED = ("name", "ingredients")


class RecipeSchemaError(ValueError):
    pass


@dataclass
class LoadReport:
    rows_in: int = 0
    rows_kept: int = 0
    dropped_no_ingredients: int = 0
    dropped_duplicates: int = 0
    warnings: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        return [f"rows in: {self.rows_in}", f"rows kept: {self.rows_kept}",
                f"dropped (no ingredients): {self.dropped_no_ingredients}",
                f"dropped (duplicate name and ingredients): {self.dropped_duplicates}",
                *[f"WARNING: {w}" for w in self.warnings]]


def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, LoadReport]:
    """Return a clean frame with ``recipe_id``, ``name``, ``ingredients_raw``, ``ingredients``, ``steps``."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise RecipeSchemaError(f"missing required columns: {missing}")
    rep = LoadReport(rows_in=len(df))
    out = pd.DataFrame({
        "recipe_id": df["id"].astype(str) if "id" in df.columns else pd.Series(range(len(df))).astype(str),
        "name": df["name"].fillna("").astype(str).str.strip(),
        "ingredients_raw": df["ingredients"].map(parse_list),
        "steps": df["steps"].map(parse_list) if "steps" in df.columns else [[] for _ in range(len(df))],
        "minutes": pd.to_numeric(df["minutes"], errors="coerce") if "minutes" in df.columns else float("nan"),
    })
    out["ingredients"] = out["ingredients_raw"].map(canonical_list)
    empty = out["ingredients"].map(len) == 0
    rep.dropped_no_ingredients = int(empty.sum())
    out = out[~empty]
    key = out["name"].str.lower() + "|" + out["ingredients"].map(lambda x: ",".join(sorted(x)))
    dup = key.duplicated()
    rep.dropped_duplicates = int(dup.sum())
    out = out[~dup].reset_index(drop=True)
    if out["recipe_id"].duplicated().any():
        rep.warnings.append("duplicate recipe ids; row numbers are used instead")
        out["recipe_id"] = out.index.astype(str)
    if "steps" not in df.columns:
        rep.warnings.append("no `steps` column: explanations cannot show the real steps")
    if out.empty:
        raise RecipeSchemaError("no recipe with ingredients is left")
    rep.rows_kept = len(out)
    return out, rep


def load_recipes(path: str | Path) -> tuple[pd.DataFrame, LoadReport]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `fridge2fork synth` or see data/README.md")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() in (".jsonl", ".json"):
        df = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    else:
        df = pd.read_csv(path)
    return prepare(df)
