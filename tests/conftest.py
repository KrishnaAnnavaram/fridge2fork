from dataclasses import replace

import pandas as pd
import pytest

from fridge2fork import synthetic
from fridge2fork.config import Settings
from fridge2fork.data import prepare
from fridge2fork.recommender import Recommender


@pytest.fixture(scope="session")
def settings():
    return replace(Settings(), dim=16, top_k=5)


@pytest.fixture(scope="session")
def recipes():
    frame, _ = prepare(synthetic.generate(400, seed=5))
    return frame


@pytest.fixture(scope="session")
def rec(recipes, settings):
    return Recommender.build(recipes, settings)


def small_frame(rows):
    return pd.DataFrame(rows, columns=["id", "name", "ingredients", "steps"])
