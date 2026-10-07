"""Ranking features, the score and the diet filters.

All scores are absolute. They do not depend on the other candidates of the same query, so a
recipe keeps its score when the candidate set changes.
"""
from __future__ import annotations

from dataclasses import dataclass

PANTRY = frozenset({"salt", "pepper", "water", "vegetable oil", "oil", "olive oil", "sugar"})

ALLERGENS: dict[str, tuple[str, ...]] = {
    "dairy": ("milk", "butter", "cheese", "cream", "yogurt", "parmesan", "mozzarella", "ricotta", "feta",
              "cheddar", "ghee", "buttermilk"),
    "egg": ("egg", "mayonnaise"),
    "gluten": ("flour", "pasta", "noodle", "bread", "couscous", "barley", "soy sauce", "tortilla", "breadcrumb"),
    "tree_nut": ("walnut", "almond", "pecan", "cashew", "pine nut", "pistachio", "hazelnut"),
    "peanut": ("peanut",),
    "shellfish": ("shrimp", "crab", "lobster", "prawn", "scallop"),
    "fish": ("salmon", "tuna", "cod", "anchovy", "fish", "tilapia"),
    "soy": ("soy", "tofu", "edamame"),
}
MEAT = ("chicken", "beef", "pork", "bacon", "ham", "sausage", "lamb", "turkey", "veal")


def _has_word(name: str, keyword: str) -> bool:
    words, kw = name.split(), keyword.split()
    return any(words[i : i + len(kw)] == kw for i in range(len(words) - len(kw) + 1))


def allergens_of(ingredients: list[str]) -> list[str]:
    """Allergen groups whose keywords appear as whole words in the ingredient names."""
    return sorted(g for g, kws in ALLERGENS.items() if any(_has_word(i, k) for i in ingredients for k in kws))


def is_vegetarian(ingredients: list[str]) -> bool:
    kws = MEAT + ALLERGENS["fish"] + ALLERGENS["shellfish"]
    return not any(_has_word(i, k) for i in ingredients for k in kws)


@dataclass(frozen=True)
class Weights:
    coverage: float = 0.5  # share of the recipe that the user has
    use: float = 0.25  # share of the user's ingredients that the recipe uses
    dense: float = 0.25  # cosine similarity, mapped from [-1, 1] to [0, 1]
    missing: float = 0.1  # penalty per missing ingredient, up to 10 missing items


@dataclass
class Features:
    matched: list[str]
    missing: list[str]
    coverage: float
    use: float
    dense: float | None
    score: float


def score_recipe(recipe: list[str], user: list[str], dense: float | None, weights: Weights,
                 assume_pantry: bool = True) -> Features:
    user_set = set(user)
    have = user_set | (PANTRY if assume_pantry else frozenset())
    rec = list(dict.fromkeys(recipe))
    matched = [i for i in rec if i in user_set]
    missing = [i for i in rec if i not in have]
    coverage = (len(rec) - len(missing)) / len(rec) if rec else 0.0
    use = len(matched) / len(user_set) if user_set else 0.0
    d01 = 0.0 if dense is None else (float(dense) + 1) / 2
    score = (weights.coverage * coverage + weights.use * use + weights.dense * d01
             - weights.missing * min(len(missing), 10) / 10)
    return Features(matched, missing, coverage, use, dense, score)
