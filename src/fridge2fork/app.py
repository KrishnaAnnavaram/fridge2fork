"""Streamlit app (extra ``ui``). Start it with ``fridge2fork ui``.

The saved index is loaded once per server process with ``st.cache_resource``.
"""
from __future__ import annotations

import streamlit as st

from fridge2fork.config import Settings
from fridge2fork.explain import grounded_explanation, template_explanation
from fridge2fork.rank import ALLERGENS
from fridge2fork.recommender import Recommender


@st.cache_resource
def _load(index_dir: str) -> Recommender:
    return Recommender.load(index_dir, Settings.from_env())


def _llm(settings: Settings):
    if settings.llm == "openai":
        from fridge2fork.llm import OpenAIChat

        return OpenAIChat(settings.llm_model, settings.llm_base_url, settings.llm_timeout_s)
    return None


def main() -> None:
    settings = Settings.from_env()
    st.set_page_config(page_title="fridge2fork", layout="wide")
    st.title("fridge2fork")
    st.caption("Recipes from the ingredients that you have. Steps come from the recipe data.")
    rec = _load(str(settings.index_dir))
    text = st.text_input("Ingredients (comma-separated)", "tomato, pasta, garlic, basil")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("Recipes", 1, 20, settings.top_k)
    max_missing = c2.slider("Maximum missing ingredients", 0, 15, 15)
    veg = c3.checkbox("Vegetarian")
    excl = st.multiselect("Exclude allergens (keyword check)", sorted(ALLERGENS))
    if not text.strip():
        return
    res = rec.recommend(text, k=k, max_missing=max_missing, exclude_allergens=tuple(excl), vegetarian=veg)
    for w in res.warnings:
        st.warning(w)
    llm = _llm(settings)
    for r in res.recommendations:
        with st.expander(f"{r.name}  (score {r.score:.3f}, missing {len(r.missing)})"):
            exp = grounded_explanation(r, res.query.known, llm) if llm else template_explanation(r)
            st.text(exp.text)
            if exp.note:
                st.caption(exp.note)


if __name__ == "__main__":
    main()
