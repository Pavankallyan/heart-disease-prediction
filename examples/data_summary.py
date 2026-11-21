"""Print a quick summary of the committed heart-disease artifacts.

Reads ``Data/heart_clean.csv`` and ``Data/model_comparison.csv`` from the
repository root and reports the dataset shape, target balance, mean clinical
values by target, and the best model by accuracy. No new numbers are invented:
everything printed is computed from the committed files.

The helpers below are covered by ``tests/test_summary.py``; ``main()`` only
wires them together and prints the rendered text.

Usage (from the repository root)::

    python examples/data_summary.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"

SUMMARY_COLUMNS = ["age", "chol", "thalach", "oldpeak"]
TARGET_LABELS = ["yes", "no"]


def load_data(data_dir=DATA_DIR):
    """Load the committed artifacts: ``(heart_clean, model_comparison)``."""
    heart = pd.read_csv(data_dir / "heart_clean.csv")
    comparison = pd.read_csv(data_dir / "model_comparison.csv")
    return heart, comparison


def target_balance(heart):
    """Return ``{label: (count, share_percent)}`` for each target label."""
    counts = heart["target"].value_counts()
    return {
        label: (int(counts[label]), 100 * counts[label] / len(heart))
        for label in TARGET_LABELS
    }


def mean_by_target(heart, columns=SUMMARY_COLUMNS):
    """Mean of each column in ``columns``, grouped by target label."""
    return heart.groupby("target")[list(columns)].mean()


def best_model(comparison):
    """The comparison-table row of the model with the highest accuracy."""
    return comparison.loc[comparison["Accuracy"].idxmax()]


def render_summary(heart, comparison):
    """Render the full summary text printed by the script."""
    lines = [f"heart_clean.csv: {len(heart)} rows x {len(heart.columns)} columns"]
    for label, (count, share) in target_balance(heart).items():
        lines.append(f"  target={label}: {count} ({share:.1f}%)")
    lines.append("Mean values by target:")
    summary = mean_by_target(heart)
    for label in TARGET_LABELS:
        row = summary.loc[label]
        lines.append(
            f"  {label}: age={row['age']:.1f}, chol={row['chol']:.1f}, "
            f"thalach={row['thalach']:.1f}, oldpeak={row['oldpeak']:.2f}"
        )
    best = best_model(comparison)
    lines.append(
        f"Best model by accuracy: {best['Model']} "
        f"(accuracy={best['Accuracy']:.4f}, AUC={best['AUC']:.4f})"
    )
    return "\n".join(lines)


def main() -> None:
    heart, comparison = load_data()
    print(render_summary(heart, comparison))


if __name__ == "__main__":
    main()
