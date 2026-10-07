"""Problem 5 (case mismatch) and problem 7 (fragile parsing)."""
import pandas as pd
import pytest

from fridge2fork.data import RecipeSchemaError, load_recipes, prepare
from fridge2fork.normalize import canonical, canonical_list, match_query, parse_list, singular


@pytest.mark.parametrize("raw, expected", [
    ("2 large Tomatoes, chopped", "tomato"),
    ("1 can garbanzo beans, drained", "chickpea"),
    ("2 tbsp extra virgin olive oil", "olive oil"),
    ("3 cloves garlic, minced", "garlic"),
    ("Scallions", "green onion"),
    ("black pepper", "pepper"),
    ("1/2 cup all-purpose flour", "flour"),
    ("fresh basil leaves", "basil leaf"),
    ("berries", "berry"),
    ("1 (14 ounce) can coconut milk", "coconut milk"),
    ("asparagus", "asparagus"),
    ("1 1/2 cups", ""),
])
def test_canonical(raw, expected):
    assert canonical(raw) == expected


def test_singular_rules():
    assert singular("potatoes") == "potato" and singular("peaches") == "peach" and singular("glass") == "glass"


def test_parse_list_handles_quotes_and_commas():
    assert parse_list("['baker\\'s chocolate', 'salt, kosher']") == ["baker's chocolate", "salt, kosher"]
    assert parse_list('["a", "b"]') == ["a", "b"]
    assert parse_list("tomato, basil; garlic") == ["tomato", "basil", "garlic"]
    assert parse_list(None) == [] and parse_list("nan") == [] and parse_list(["x", ""]) == ["x"]


def test_canonical_list_removes_duplicates():
    assert canonical_list("['2 tomatoes', 'tomato', 'Salt']") == ["tomato", "salt"]


def test_query_matching_corrects_close_spellings_and_reports_unknown():
    m = match_query("Tomatos, mozarella, unobtainium, Garlic", {"tomato", "mozzarella", "garlic"})
    assert m.known == ["tomato", "mozzarella", "garlic"]
    assert m.corrections == {"mozarella": "mozzarella"}  # "Tomatos" becomes "tomato" by the plural rule
    assert m.unknown == ["unobtainium"]


def test_prepare_validates_and_cleans():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4],
        "name": ["A", "A", "B", "C"],
        "ingredients": ["['2 Tomatoes']", "['tomato']", "[]", "['egg']"],
        "steps": ["['Cut.']", "['Cut.']", "[]", "['Boil.']"],
    })
    out, rep = prepare(df)
    assert list(out["name"]) == ["A", "C"]
    assert rep.dropped_no_ingredients == 1 and rep.dropped_duplicates == 1
    assert out.loc[0, "ingredients"] == ["tomato"] and out.loc[0, "ingredients_raw"] == ["2 Tomatoes"]
    with pytest.raises(RecipeSchemaError):
        prepare(pd.DataFrame({"name": ["x"]}))


def test_prepare_warns_without_steps():
    _, rep = prepare(pd.DataFrame({"name": ["x"], "ingredients": ["['egg']"]}))
    assert any("steps" in w for w in rep.warnings)


def test_load_recipes_formats(tmp_path):
    df = pd.DataFrame({"name": ["x"], "ingredients": ["['egg', 'milk']"]})
    df.to_csv(tmp_path / "r.csv", index=False)
    df.to_json(tmp_path / "r.jsonl", orient="records", lines=True)
    assert len(load_recipes(tmp_path / "r.csv")[0]) == 1
    assert len(load_recipes(tmp_path / "r.jsonl")[0]) == 1
    with pytest.raises(FileNotFoundError, match="fridge2fork synth"):
        load_recipes(tmp_path / "none.csv")
