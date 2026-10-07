"""Problems 1 (zero query vector), 2 (unstable score), 3 (low recall), 5 (missing list) and 6 (cold start)."""
import json
from dataclasses import replace

import numpy as np
import pytest

from fridge2fork.data import prepare
from fridge2fork.embed import SVDEmbedder, make_embedder
from fridge2fork.index import InvertedIndex, NumpyIndex, make_vector_index
from fridge2fork.rank import Weights, allergens_of, is_vegetarian, score_recipe
from fridge2fork.recommender import Recommender

from .conftest import small_frame


def test_unknown_only_query_gives_no_recommendation(rec):
    res = rec.recommend("unobtainium, kryptonite")
    assert res.recommendations == []
    assert any("no known ingredient" in w for w in res.warnings)
    assert res.query.unknown == ["unobtainium", "kryptonite"]


def test_query_vector_is_never_zero(rec):
    assert rec.embedder.vector(["not an ingredient"]) is None
    v = rec.embedder.vector(["tomato", "garlic"])
    assert np.isclose(np.linalg.norm(v), 1.0)


def test_recipe_without_vector_is_never_a_dense_hit():
    vecs = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    ids, sims = NumpyIndex(vecs, np.array([True, False])).search(np.array([0.0, 1.0]), 5)
    assert ids.tolist() == [0]


def test_score_is_absolute_and_finite():
    w = Weights()
    exact = score_recipe(["tomato", "basil"], ["tomato", "basil"], 1.0, w)
    assert np.isfinite(exact.score) and exact.coverage == 1.0 and exact.missing == []
    # the same recipe gets the same score in any candidate set: score_recipe sees only this recipe
    again = score_recipe(["tomato", "basil"], ["tomato", "basil"], 1.0, w)
    assert again.score == exact.score
    worse = score_recipe(["tomato", "basil", "cream", "egg"], ["tomato", "basil"], 1.0, w)
    assert worse.score < exact.score and worse.missing == ["cream", "egg"]


def test_pantry_items_are_not_missing():
    f = score_recipe(["pasta", "salt", "olive oil"], ["pasta"], None, Weights(), assume_pantry=True)
    assert f.missing == [] and f.use == 1.0
    g = score_recipe(["pasta", "salt"], ["pasta"], None, Weights(), assume_pantry=False)
    assert g.missing == ["salt"]


def test_missing_uses_normalised_names(settings):
    frame, _ = prepare(small_frame([
        (1, "Salad", "['2 large Tomatoes, chopped', '1 Cucumber', 'Kosher salt']", "['Cut.', 'Mix.']"),
        (2, "Soup", "['3 potatoes', 'onion']", "['Boil.']"),
        (3, "Pasta", "['pasta', 'tomato']", "['Boil.']"),
    ]))
    r = Recommender.build(frame, replace(settings, min_count=1, dim=2))
    top = r.recommend("Tomatoes, cucumbers").recommendations[0]
    assert top.name == "Salad" and top.missing == [] and top.matched == ["tomato", "cucumber"]


def test_overlap_candidates_find_recipes_far_from_the_average(settings):
    # 60 recipes share the query items with many others; the target has all of them plus rare items
    rows = [(i, f"R{i}", repr(["tomato", "pasta", f"x{i % 5}", f"y{i % 7}"]), "['Cook.']") for i in range(60)]
    rows.append((999, "Target", repr(["tomato", "pasta", "basil", "garlic", "anchovy", "caper", "olive"]), "['Cook.']"))
    frame, _ = prepare(small_frame(rows))
    s = replace(settings, min_count=1, dim=4, dense_k=3, overlap_k=50)
    r = Recommender.build(frame, s)
    res = r.recommend("tomato, pasta, basil, garlic, anchovy, caper, olive", k=3)
    assert res.recommendations[0].name == "Target"


def test_filters(rec):
    res = rec.recommend("pasta, tomato, garlic, parmesan", k=20, exclude_allergens=("dairy", "gluten"))
    assert all(not {"dairy", "gluten"} & set(r.allergens) for r in res.recommendations)
    veg = rec.recommend("chicken, rice, onion", k=20, vegetarian=True)
    assert all(is_vegetarian(r.matched + r.missing) for r in veg.recommendations)
    few = rec.recommend("pasta, tomato", k=20, max_missing=1)
    assert all(len(r.missing) <= 1 for r in few.recommendations)


def test_allergen_keywords_match_whole_words():
    assert allergens_of(["peanut butter", "egg"]) == ["dairy", "egg", "peanut"]
    assert allergens_of(["eggplant"]) == []
    assert not is_vegetarian(["chicken"]) and is_vegetarian(["eggplant", "rice"])


def test_save_and_load_give_the_same_results(rec, tmp_path, settings):
    folder = rec.save(tmp_path / "idx")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["recipes"] == len(rec.recipes) and manifest["embedder"] == "svd"
    loaded = Recommender.load(folder, settings)
    a = rec.recommend("tomato, basil, pasta")
    b = loaded.recommend("tomato, basil, pasta")
    assert [x.recipe_id for x in a.recommendations] == [x.recipe_id for x in b.recommendations]
    manifest["format_version"] = 0
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="format version"):
        Recommender.load(folder, settings)
    with pytest.raises(FileNotFoundError):
        Recommender.load(tmp_path / "none", settings)


def test_embeddings_are_seeded(recipes):
    lists = recipes["ingredients"].tolist()
    a = SVDEmbedder(dim=8, seed=1).fit(lists).vectors
    b = SVDEmbedder(dim=8, seed=1).fit(lists).vectors
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        SVDEmbedder(min_count=10_000).fit(lists)
    with pytest.raises(ValueError):
        make_embedder("glove", 8, 1, 1)


def test_inverted_index_counts_overlap():
    inv = InvertedIndex([["a", "b"], ["b", "c"], ["d"]], ["a", "b", "c", "d"])
    assert inv.overlap(["b", "c", "zzz"]).tolist() == [1, 2, 0]
    assert inv.top(["b", "c"], 1).tolist() == [1]


def test_faiss_index_matches_numpy(rec):
    pytest.importorskip("faiss")
    q = rec.embedder.vector(["tomato", "basil"])
    a, _ = make_vector_index("numpy", rec.vectors, rec.valid).search(q, 10)
    b, _ = make_vector_index("faiss", rec.vectors, rec.valid).search(q, 10)
    assert set(a.tolist()) == set(b.tolist())


def test_word2vec_embedder_optional(recipes):
    pytest.importorskip("gensim")
    emb = make_embedder("word2vec", 8, 1, 2).fit(recipes["ingredients"].tolist())
    assert emb.vectors.shape[1] == 8
