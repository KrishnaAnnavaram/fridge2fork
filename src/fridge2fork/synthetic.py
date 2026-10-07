"""Synthetic recipes in the Food.com layout, for the demo and the tests (no download).

Each recipe takes most ingredients from one cuisine pool and a few pantry items. The raw
ingredient strings have quantities, plurals and descriptors, so the normaliser has work to do.
All recipe names and steps are made up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CUISINES: dict[str, list[str]] = {
    "italian": ["pasta", "tomato", "garlic", "basil", "olive oil", "parmesan", "mozzarella", "oregano",
                "onion", "zucchini", "mushroom", "spinach", "ricotta", "pine nut", "lemon"],
    "mexican": ["tortilla", "black bean", "corn", "avocado", "lime", "cilantro", "jalapeno", "onion",
                "tomato", "cumin", "cheddar", "chicken", "bell pepper", "sour cream", "rice"],
    "indian": ["chickpea", "lentil", "rice", "onion", "garlic", "ginger", "turmeric", "cumin", "garam masala",
               "coconut milk", "spinach", "potato", "cauliflower", "yogurt", "cilantro"],
    "asian": ["soy sauce", "ginger", "garlic", "green onion", "rice", "noodle", "sesame oil", "tofu", "egg",
              "broccoli", "carrot", "shrimp", "chili", "peanut", "bok choy"],
    "baking": ["flour", "sugar", "butter", "egg", "milk", "baking powder", "vanilla", "cinnamon", "brown sugar",
               "walnut", "chocolate chip", "banana", "oat", "honey", "baking soda"],
    "salad": ["lettuce", "cucumber", "tomato", "feta", "olive", "red onion", "lemon", "olive oil", "quinoa",
              "chickpea", "avocado", "walnut", "apple", "spinach", "carrot"],
}
PANTRY = ["salt", "pepper", "water", "vegetable oil"]
RAW_FORM = {  # how a canonical name can appear in a raw recipe
    "tomato": ["2 large tomatoes, chopped", "1 cup diced tomatoes", "tomato"],
    "onion": ["1 medium onion, diced", "2 onions", "onion"],
    "garlic": ["3 cloves garlic, minced", "garlic"],
    "egg": ["2 eggs, beaten", "1 large egg", "egg"],
    "flour": ["2 cups all-purpose flour", "flour"],
    "sugar": ["1 cup granulated sugar", "sugar"],
    "pepper": ["black pepper", "pepper to taste"],
    "salt": ["kosher salt", "1 tsp salt"],
    "potato": ["3 potatoes, peeled and cubed", "potato"],
    "green onion": ["2 scallions, sliced", "green onions"],
    "chickpea": ["1 can garbanzo beans, drained", "chickpeas"],
    "olive oil": ["2 tbsp extra virgin olive oil", "olive oil"],
}
VERBS = ["Heat", "Mix", "Stir in", "Add", "Combine", "Toss", "Fold in", "Season with"]


def generate(n_recipes: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = list(CUISINES)
    rows = []
    for i in range(n_recipes):
        cuisine = names[int(rng.integers(len(names)))]
        pool = CUISINES[cuisine]
        k = int(rng.integers(4, 9))
        main = [str(x) for x in rng.choice(pool, size=k, replace=False)]
        if rng.random() < 0.3:  # a few cross-cuisine items
            other = CUISINES[names[int(rng.integers(len(names)))]]
            main.append(str(rng.choice(other)))
        pantry = [str(x) for x in rng.choice(PANTRY, size=int(rng.integers(0, 3)), replace=False)]
        ingredients = list(dict.fromkeys(main + pantry))
        raw = [str(rng.choice(RAW_FORM[x])) if x in RAW_FORM else x for x in ingredients]
        title = f"{cuisine.title()} {ingredients[0]} and {ingredients[1]} dish {i}"
        steps = [f"{VERBS[j % len(VERBS)]} the {ing}." for j, ing in enumerate(ingredients[:5])]
        steps.append(f"Cook for {int(rng.integers(10, 50))} minutes and serve.")
        rows.append({
            "id": 100000 + i, "name": title, "minutes": int(rng.integers(10, 90)),
            "ingredients": repr(raw), "steps": repr(steps), "cuisine": cuisine,
        })
    return pd.DataFrame(rows)
