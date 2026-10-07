"""Explanations of a recommendation. The steps always come from the recipe data.

* ``template_explanation``: deterministic text from the ranking features, a substitution
  table and the real steps of the recipe. It needs no model.
* ``grounded_explanation``: asks a chat model to explain the match and to suggest substitutions
  for the missing ingredients. The prompt holds the real ingredients and steps. The model must
  not write steps. The reply is refused, and the template is used, if it is empty, if it has a
  number that is not in the recipe, or if it lists numbered steps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import ChatModel
from .recommender import Recommendation

SUBSTITUTES: dict[str, list[str]] = {
    "butter": ["olive oil", "margarine"], "milk": ["oat milk", "soy milk", "water and a little butter"],
    "egg": ["1 tbsp ground flaxseed in 3 tbsp water"], "sour cream": ["plain yogurt"],
    "yogurt": ["sour cream"], "lemon": ["lime", "white vinegar"], "lime": ["lemon"],
    "cilantro": ["parsley"], "basil": ["parsley", "spinach"], "parmesan": ["pecorino", "nutritional yeast"],
    "buttermilk": ["milk with 1 tbsp lemon juice"], "brown sugar": ["sugar with a little honey"],
    "honey": ["maple syrup", "sugar"], "soy sauce": ["tamari", "salt"], "shallot": ["onion"],
    "red onion": ["onion"], "green onion": ["onion", "chive"], "ricotta": ["cottage cheese"],
    "mozzarella": ["cheddar"], "feta": ["goat cheese"], "rice": ["quinoa", "couscous"],
    "pasta": ["noodle", "rice"], "noodle": ["pasta"], "chicken": ["tofu", "chickpea"],
    "shrimp": ["chicken", "tofu"], "tortilla": ["flatbread"], "baking powder": ["baking soda with lemon juice"],
}
SYSTEM = (
    "You explain why a recipe fits the ingredients that a user has. Use only the recipe data that you get. "
    "Do not write cooking steps, cooking times or temperatures. Do not invent ingredients. "
    "Give two short parts: 'Why it fits' and 'Substitutions' for the missing ingredients."
)


@dataclass
class Explanation:
    text: str
    source: str  # "template" or the model name
    note: str = ""


def substitutions(missing: list[str]) -> dict[str, list[str]]:
    return {m: SUBSTITUTES[m] for m in missing if m in SUBSTITUTES}


def _steps(rec: Recommendation) -> str:
    if not rec.steps:
        return "Steps: the recipe data has no steps for this recipe."
    return "Steps (from the recipe data):\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(rec.steps, 1))


def template_explanation(rec: Recommendation) -> Explanation:
    lines = [f"{rec.name}: you have {len(rec.matched)} of its ingredients "
             f"({rec.coverage:.0%} of the recipe, pantry items included)."]
    if rec.matched:
        lines.append("Matched: " + ", ".join(rec.matched) + ".")
    if rec.missing:
        lines.append("Missing: " + ", ".join(rec.missing) + ".")
        subs = substitutions(rec.missing)
        for m, alts in subs.items():
            lines.append(f"Instead of {m}, you can use {' or '.join(alts)}.")
    else:
        lines.append("Nothing is missing.")
    if rec.allergens:
        lines.append("Allergen warning (keyword check, read the labels): " + ", ".join(rec.allergens) + ".")
    lines.append(_steps(rec))
    return Explanation(text="\n".join(lines), source="template")


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,/]\d+)?", text))


def check_reply(reply: str, rec: Recommendation) -> str | None:
    """Return a reason to refuse the reply, or ``None`` if it is acceptable."""
    if not reply.strip():
        return "empty reply"
    allowed = _numbers(" ".join(rec.ingredients_raw + rec.steps + [rec.name]))
    allowed |= _numbers(" ".join(a for alts in SUBSTITUTES.values() for a in alts))
    extra = _numbers(reply) - allowed
    if extra:
        return f"numbers not in the recipe: {sorted(extra)}"
    if len(re.findall(r"(?m)^\s*(?:step\s*)?\d+[.)]\s", reply, flags=re.I)) >= 2:
        return "the reply lists steps"
    return None


def grounded_explanation(rec: Recommendation, user_ingredients: list[str], llm: ChatModel) -> Explanation:
    user = (
        f"Recipe: {rec.name}\n"
        f"Recipe ingredients: {'; '.join(rec.ingredients_raw)}\n"
        f"Recipe steps (do not repeat or change them): {' | '.join(rec.steps) or 'none'}\n"
        f"User has: {', '.join(user_ingredients)}\n"
        f"Matched: {', '.join(rec.matched) or 'none'}\n"
        f"Missing: {', '.join(rec.missing) or 'none'}\n"
        f"Known substitutions: {substitutions(rec.missing) or 'none'}"
    )
    try:
        reply = llm.complete(SYSTEM, user)
    except Exception as exc:  # noqa: BLE001 - any provider error falls back to the template
        base = template_explanation(rec)
        return Explanation(base.text, "template", f"model error: {type(exc).__name__}")
    reason = check_reply(reply, rec)
    if reason:
        base = template_explanation(rec)
        return Explanation(base.text, "template", f"model reply refused: {reason}")
    return Explanation(reply.strip() + "\n\n" + _steps(rec), llm.name)
