"""Shared fixtures for the heart-disease-prediction data tests.

The repository ships its analysis artifacts as committed files, so the tests
read them from the repo-relative ``Data/`` directory — the same layout CI and
contributors see.
"""
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"


@pytest.fixture(scope="session")
def heart_clean():
    """The cleaned dataset written out by the R analysis script."""
    return pd.read_csv(DATA_DIR / "heart_clean.csv")


@pytest.fixture(scope="session")
def model_comparison():
    """The trained-model comparison table committed in Data/."""
    return pd.read_csv(DATA_DIR / "model_comparison.csv")
