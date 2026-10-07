"""Problems 4 (invented steps), 8 (no offline evaluation) and 9 (legacy SDK and keys)."""
from dataclasses import replace

import pytest

from fridge2fork import evaluate as ev
from fridge2fork.cli import main
from fridge2fork.config import Settings, read_dotenv
from fridge2fork.explain import check_reply, grounded_explanation, template_explanation
from fridge2fork.llm import ScriptedLLM
from fridge2fork.recommender import Recommendation


def _rec():
    return Recommendation(
        recipe_id="1", name="Tomato pasta", score=0.9, coverage=0.75, use=1.0, dense=0.8,
        matched=["pasta", "tomato"], missing=["basil", "butter"], allergens=["gluten"], minutes=20.0,
        steps=["Boil the pasta for 10 minutes.", "Add the tomato."], ingredients_raw=["200 g pasta", "2 tomatoes", "basil", "butter"])


def test_template_uses_real_steps_and_substitutions():
    e = template_explanation(_rec())
    assert "1. Boil the pasta for 10 minutes." in e.text and "2. Add the tomato." in e.text
    assert "Instead of butter, you can use olive oil or margarine." in e.text
    assert "gluten" in e.text and e.source == "template"


def test_grounded_prompt_has_real_steps_and_reply_is_kept():
    llm = ScriptedLLM(["Why it fits: you have the pasta and the tomato.\nSubstitutions: olive oil for butter."])
    e = grounded_explanation(_rec(), ["pasta", "tomato"], llm)
    system, user = llm.prompts[0]
    assert "Do not write cooking steps" in system
    assert "Boil the pasta for 10 minutes." in user
    assert e.source == "scripted" and "Steps (from the recipe data)" in e.text


@pytest.mark.parametrize("reply, reason", [
    ("", "empty"),
    ("Bake at 220 degrees for 45 minutes.", "numbers not in the recipe"),
    ("1. Boil pasta.\n2. Add sauce.\n", "lists steps"),
])
def test_bad_replies_fall_back_to_the_template(reply, reason):
    assert reason in check_reply(reply, _rec())
    e = grounded_explanation(_rec(), ["pasta"], ScriptedLLM([reply]))
    assert e.source == "template" and "refused" in e.note


def test_numbers_from_the_recipe_are_allowed():
    assert check_reply("Boil it for 10 minutes, as the recipe says.", _rec()) is None


def test_model_errors_fall_back():
    class Broken:
        name = "broken"

        def complete(self, system, user):
            raise TimeoutError("slow")

    e = grounded_explanation(_rec(), ["pasta"], Broken())
    assert e.source == "template" and "TimeoutError" in e.note


def test_openai_adapter_needs_a_key(monkeypatch):
    from fridge2fork.llm import OpenAIChat

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        OpenAIChat("gpt-4o-mini")


def test_evaluation_holds_out_recipes_from_the_embedder(recipes, settings, monkeypatch):
    seen = {}
    real = ev.Recommender.build

    def spy(rec_frame, s, fit_on=None, weights=None):
        seen["fit_ids"] = set(fit_on["recipe_id"])
        return real(rec_frame, s, fit_on=fit_on, weights=weights)

    monkeypatch.setattr(ev.Recommender, "build", staticmethod(spy))
    table = ev.evaluate(recipes, settings, n_queries=40, hide=2)
    q = ev.make_queries(recipes, 40, 2, settings.seed)
    assert not seen["fit_ids"] & set(q["recipe_id"])
    assert list(table["variant"]) == ["dense", "overlap", "hybrid"]
    assert ((table["recall@1"] <= table["recall@5"]) & (table["recall@5"] <= table["recall@10"])).all()


def test_queries_hide_the_requested_number(recipes):
    q = ev.make_queries(recipes, 10, 3, seed=1)
    full = recipes.set_index("recipe_id")["ingredients"]
    for _, row in q.iterrows():
        assert len(row["hidden"]) == 3
        assert set(row["query"]) | set(row["hidden"]) == set(full[row["recipe_id"]])
    with pytest.raises(ValueError):
        ev.make_queries(recipes, 10, 50, seed=1)


def test_settings(monkeypatch, tmp_path):
    s = Settings.from_env({"FRIDGE2FORK_DIM": "32", "FRIDGE2FORK_MAX_MISSING": "2", "FRIDGE2FORK_ASSUME_PANTRY": "no"})
    assert s.dim == 32 and s.max_missing == 2 and s.assume_pantry is False
    for bad in ({"FRIDGE2FORK_EMBEDDER": "glove"}, {"FRIDGE2FORK_LLM": "other"}, {"FRIDGE2FORK_DIM": "1"}):
        with pytest.raises(ValueError):
            Settings.from_env(bad)
    (tmp_path / ".env").write_text("FRIDGE2FORK_TOP_K=7\nOPENAI_API_KEY=\n", encoding="utf-8")
    assert read_dotenv(tmp_path / ".env") == {"FRIDGE2FORK_TOP_K": "7"}


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("FRIDGE2FORK_DIM", "16")
    data = tmp_path / "r.csv"
    idx = tmp_path / "idx"
    assert main(["synth", "--out", str(data), "-n", "300"]) == 0
    assert main(["build", "--recipes", str(data), "--index-dir", str(idx)]) == 0
    assert main(["recommend", "mozarella, pasta, garlic", "--index-dir", str(idx), "-k", "2", "--explain"]) == 0
    out = capsys.readouterr().out
    assert "corrected: mozarella -> mozzarella" in out and "Steps (from the recipe data)" in out
    assert main(["recommend", "pasta", "--index-dir", str(idx), "--json"]) == 0
    assert '"recommendations"' in capsys.readouterr().out
    assert main(["evaluate", "--recipes", str(data), "--queries", "20", "--out", str(tmp_path / "e.csv")]) == 0
    assert (tmp_path / "e.csv").exists()


def test_cli_errors(tmp_path, capsys):
    assert main(["recommend", "egg", "--index-dir", str(tmp_path / "none")]) == 2
    assert "fridge2fork build" in capsys.readouterr().err
