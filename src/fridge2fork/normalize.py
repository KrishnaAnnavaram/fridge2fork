"""Ingredient normalisation: one canonical name per ingredient.

``"2 large Tomatoes, chopped"`` and ``"tomato"`` give the same name, ``"tomato"``. The same
function runs on the recipe data and on the user query, so the two sides always match.
"""
from __future__ import annotations

import ast
import difflib
import json
import re
from dataclasses import dataclass, field

UNITS = {
    "cup", "cups", "c", "tablespoon", "tablespoons", "tbsp", "tbs", "teaspoon", "teaspoons", "tsp", "ounce",
    "ounces", "oz", "pound", "pounds", "lb", "lbs", "gram", "grams", "g", "kg", "kilogram", "kilograms", "ml",
    "l", "liter", "liters", "litre", "litres", "pinch", "dash", "can", "cans", "package", "packages", "pkg",
    "clove", "cloves", "slice", "slices", "stick", "sticks", "bunch", "handful", "piece", "pieces", "quart",
    "pint", "jar", "bottle", "head", "sprig", "sprigs",
}
DESCRIPTORS = {
    "chopped", "diced", "minced", "sliced", "fresh", "freshly", "large", "small", "medium", "ground", "grated",
    "shredded", "crushed", "finely", "roughly", "thinly", "peeled", "seeded", "cooked", "uncooked", "raw",
    "frozen", "canned", "dried", "boneless", "skinless", "to", "taste", "optional", "softened", "melted",
    "room", "temperature", "beaten", "whole", "halved", "cubed", "rinsed", "drained", "packed", "divided",
    "of", "and", "or", "a", "an", "the", "for", "about", "plus", "more", "extra", "lightly", "cold",
    "warm", "organic", "lean", "trimmed",
}
IRREGULAR = {
    "leaves": "leaf", "loaves": "loaf", "halves": "half", "knives": "knife", "potatoes": "potato",
    "tomatoes": "tomato", "mangoes": "mango", "avocados": "avocado", "anchovies": "anchovy",
    "chilies": "chili", "chillies": "chili", "radishes": "radish", "peaches": "peach", "dishes": "dish",
    "molasses": "molasses", "hummus": "hummus", "couscous": "couscous", "asparagus": "asparagus",
    "citrus": "citrus", "swiss": "swiss", "grass": "grass", "lettuce": "lettuce", "cheese": "cheese",
    "rice": "rice", "juice": "juice", "sauce": "sauce", "spice": "spice", "molass": "molasses",
    "oats": "oat", "greens": "greens", "noodles": "noodle", "chickpeas": "chickpea", "lentils": "lentil",
}
SYNONYMS = {
    "scallion": "green onion", "spring onion": "green onion", "capsicum": "bell pepper",
    "coriander leaf": "cilantro", "garbanzo bean": "chickpea", "garbanzo": "chickpea", "aubergine": "eggplant",
    "courgette": "zucchini", "prawn": "shrimp", "beef mince": "beef",
    "caster sugar": "sugar", "granulated sugar": "sugar", "white sugar": "sugar", "plain flour": "flour",
    "all-purpose flour": "flour", "all purpose flour": "flour", "extra virgin olive oil": "olive oil",
    "virgin olive oil": "olive oil", "kosher salt": "salt", "sea salt": "salt", "table salt": "salt",
    "black pepper": "pepper", "egg white": "egg", "egg yolk": "egg", "unsalted butter": "butter",
    "salted butter": "butter", "chilli": "chili", "chile": "chili", "parmesan cheese": "parmesan",
}
_NUM = re.compile(r"^[\d./¼-¾⅐-⅞-]+$")


def singular(word: str) -> str:
    if word in IRREGULAR:
        return IRREGULAR[word]
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "xes", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def canonical(raw: str) -> str:
    """Return the canonical ingredient name, or ``""`` if nothing is left."""
    text = raw.lower()
    text = re.sub(r"\(.*?\)", " ", text)  # remove notes in brackets
    text = text.split(",")[0]  # "tomatoes, chopped" -> "tomatoes"
    text = re.sub(r"[^a-z0-9\s/.\-¼-¾⅐-⅞]", " ", text)
    words = [w for w in text.split() if w and not _NUM.match(w) and w not in UNITS and w not in DESCRIPTORS]
    words = [singular(w) for w in words]
    name = " ".join(words).strip()
    return SYNONYMS.get(name, name)


def parse_list(value) -> list[str]:
    """Parse a list-like cell: a Python or JSON list string, a real list, or comma-separated text."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("["):
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(text)
                if isinstance(out, list):
                    return [str(v) for v in out if str(v).strip()]
            except (ValueError, SyntaxError):
                continue
    return [p.strip() for p in re.split(r"[,;\n]", text) if p.strip()]


def canonical_list(values) -> list[str]:
    """Canonical names without duplicates, in first-seen order."""
    seen: dict[str, None] = {}
    for v in parse_list(values):
        c = canonical(v)
        if c:
            seen.setdefault(c, None)
    return list(seen)


@dataclass
class QueryMatch:
    known: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)  # user text -> vocabulary name
    unknown: list[str] = field(default_factory=list)


def match_query(text_or_list, vocabulary: set[str], cutoff: float = 0.85) -> QueryMatch:
    """Map the user ingredients to the vocabulary. Close spellings are corrected. The rest is unknown."""
    out = QueryMatch()
    vocab = sorted(vocabulary)
    for raw in parse_list(text_or_list):
        c = canonical(raw)
        if not c:
            continue
        if c in vocabulary:
            name = c
        else:
            close = difflib.get_close_matches(c, vocab, n=1, cutoff=cutoff)
            if not close:
                out.unknown.append(raw.strip())
                continue
            name = close[0]
            out.corrections[raw.strip()] = name
        if name not in out.known:
            out.known.append(name)
    return out
