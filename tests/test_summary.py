"""Unit tests for the examples/data_summary.py helpers.

The example script only reports numbers computed from the committed files,
so these tests pin its helpers against those same files: shapes, the target
balance, the mean clinical values by target, the best-model pick, and the
exact rendered text.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.data_summary import (  # noqa: E402
    best_model,
    load_data,
    mean_by_target,
    render_summary,
    target_balance,
)


def test_load_data_shapes(heart_clean, model_comparison):
    heart, comparison = load_data()
    assert len(heart) == 1025
    assert len(heart.columns) == 14
    assert len(comparison) == 2
    assert list(comparison.columns) == [
        "Model", "Accuracy", "Sensitivity", "Specificity", "AUC",
    ]
    # The loaded frames are the same artifacts the fixtures read.
    assert heart.equals(heart_clean)
    assert comparison.equals(model_comparison)


def test_target_balance_counts_and_shares(heart_clean):
    balance = target_balance(heart_clean)
    assert balance["yes"] == (526, pytest.approx(51.317, abs=0.01))
    assert balance["no"] == (499, pytest.approx(48.683, abs=0.01))
    assert sum(share for _, share in balance.values()) == pytest.approx(100.0)


def test_mean_by_target_values(heart_clean):
    means = mean_by_target(heart_clean)
    assert list(means.index) == ["no", "yes"]  # groupby sorts the labels
    assert list(means.columns) == ["age", "chol", "thalach", "oldpeak"]
    yes = means.loc["yes"]
    assert yes["age"] == pytest.approx(52.41, abs=0.01)
    assert yes["chol"] == pytest.approx(240.98, abs=0.01)
    assert yes["thalach"] == pytest.approx(158.59, abs=0.01)
    assert yes["oldpeak"] == pytest.approx(0.57, abs=0.01)
    no = means.loc["no"]
    assert no["age"] == pytest.approx(56.57, abs=0.01)
    assert no["chol"] == pytest.approx(251.29, abs=0.01)
    assert no["thalach"] == pytest.approx(139.13, abs=0.01)
    assert no["oldpeak"] == pytest.approx(1.60, abs=0.01)
    # Disease-present patients are younger with higher max heart rate here.
    assert yes["age"] < no["age"]
    assert yes["thalach"] > no["thalach"]


def test_mean_by_target_custom_columns(heart_clean):
    means = mean_by_target(heart_clean, columns=["age"])
    assert list(means.columns) == ["age"]
    assert means.loc["yes", "age"] == pytest.approx(52.41, abs=0.01)


def test_best_model_is_knn(model_comparison):
    best = best_model(model_comparison)
    assert best["Model"] == "KNN (K=3)"
    assert best["Accuracy"] == pytest.approx(0.8775, abs=0.0001)
    assert best["AUC"] == pytest.approx(0.9859, abs=0.0001)


def test_render_summary_exact_text(heart_clean, model_comparison):
    expected = (
        "heart_clean.csv: 1025 rows x 14 columns\n"
        "  target=yes: 526 (51.3%)\n"
        "  target=no: 499 (48.7%)\n"
        "Mean values by target:\n"
        "  yes: age=52.4, chol=241.0, thalach=158.6, oldpeak=0.57\n"
        "  no: age=56.6, chol=251.3, thalach=139.1, oldpeak=1.60\n"
        "Best model by accuracy: KNN (K=3) (accuracy=0.8775, AUC=0.9859)"
    )
    assert render_summary(heart_clean, model_comparison) == expected


def test_render_summary_line_count(heart_clean, model_comparison):
    lines = render_summary(heart_clean, model_comparison).splitlines()
    assert len(lines) == 7
    assert lines[0].startswith("heart_clean.csv: 1025 rows")
    assert lines[-1].startswith("Best model by accuracy: KNN (K=3)")
