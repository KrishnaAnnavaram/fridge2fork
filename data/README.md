# Data

Git does not track the files in this folder, only this README. No recipe data is in the repository.

## Synthetic recipes (default, no download)

```bash
fridge2fork synth --out data/synthetic_recipes.csv -n 1500
```

The generator (`src/fridge2fork/synthetic.py`) writes made-up recipes in the Food.com layout. The raw ingredient
strings have quantities, plurals and descriptors ("2 large tomatoes, chopped"). The tests and `fridge2fork demo`
use it.

## Food.com recipes

| Item | Value |
|---|---|
| Source | Food.com recipes and interactions (`RAW_recipes.csv`), published on Kaggle |
| URL | https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions |
| License and terms | See the Kaggle dataset page. Cite the dataset authors (Majumder et al., EMNLP 2019) if you publish results. Do not commit the file. |
| Size | About 230,000 recipes, about 280 MB as CSV |
| Expected file | `data/recipes.csv` (or set `FRIDGE2FORK_RECIPES`). CSV, JSON Lines and Parquet (needs `pyarrow`) are read |

The related Kaggle dataset "Food.com Recipes with Search Terms and Tags"
(`shuyangli94/foodcom-recipes-with-search-terms-and-tags`) has the same columns and can be used the same way.

## Columns

| Column | Required | Meaning |
|---|---|---|
| `name` | Yes | Recipe title |
| `ingredients` | Yes | A list string, for example `"['2 tomatoes', 'olive oil']"`, a JSON list, or comma-separated text |
| `id` | No | Recipe id. Row numbers are used if it is absent or not unique |
| `steps` | No | A list string of the steps. Explanations show these steps. Without it, no steps are shown |
| `minutes` | No | Preparation time |

Other columns (`tags`, `nutrition`, `description`, ...) are ignored.

## Generated files

`fridge2fork build` writes the index to `artifacts/index/` (`FRIDGE2FORK_INDEX_DIR`): `recipes.pkl`,
`ingredient_vectors.npy`, `vocabulary.json` and `manifest.json`. Git ignores this folder. Load only an index
that you built yourself, because `recipes.pkl` is a pickle file.
