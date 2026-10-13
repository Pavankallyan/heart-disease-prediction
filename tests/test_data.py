"""Validation tests for the committed data artifacts.

These guard the outputs of the R analysis script (``Code/``): the cleaned
dataset it writes and the model comparison table. If a future edit to the
script breaks the cleaning or modeling steps, these fail before the bad
artifacts get committed.
"""

EXPECTED_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]


def test_heart_clean_schema(heart_clean):
    assert list(heart_clean.columns) == EXPECTED_COLUMNS
    assert len(heart_clean) > 1000


def test_heart_clean_no_missing_values(heart_clean):
    assert heart_clean.isna().sum().sum() == 0


def test_target_is_binary_yes_no(heart_clean):
    assert set(heart_clean["target"].unique()) == {"yes", "no"}
    assert (heart_clean["target"] == "yes").mean() > 0.4  # not degenerate
    assert (heart_clean["target"] == "yes").mean() < 0.6


def test_clinical_fields_in_sane_ranges(heart_clean):
    assert heart_clean["age"].between(20, 100).all()
    assert heart_clean["trestbps"].between(50, 250).all()
    assert heart_clean["chol"].between(50, 700).all()
    assert heart_clean["thalach"].between(50, 250).all()
    assert (heart_clean["oldpeak"] >= 0).all()
    assert heart_clean["ca"].between(0, 4).all()


def test_model_comparison_covers_both_models(model_comparison):
    assert set(model_comparison["Model"]) == {
        "Logistic Regression", "KNN (K=3)",
    }


def test_model_metrics_are_valid_scores(model_comparison):
    for metric in ["Accuracy", "Sensitivity", "Specificity", "AUC"]:
        scores = model_comparison[metric]
        assert (scores > 0.5).all()   # better than a coin flip
        assert (scores <= 1.0).all()   # proper probability-scale scores


def test_knn_matches_reported_best_run(model_comparison):
    knn = model_comparison.set_index("Model").loc["KNN (K=3)"]
    assert knn["Accuracy"] > 0.87
    assert knn["AUC"] > 0.98
