"""Print a quick summary of the committed heart-disease artifacts.

Reads ``Data/heart_clean.csv`` and ``Data/model_comparison.csv`` from the
repository root and reports the dataset shape, target balance, mean clinical
values by target, and the best model by accuracy. No new numbers are invented:
everything printed is computed from the committed files.

Usage (from the repository root)::

    python examples/data_summary.py
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"


def main() -> None:
    heart = pd.read_csv(DATA_DIR / "heart_clean.csv")
    comparison = pd.read_csv(DATA_DIR / "model_comparison.csv")

    print(f"heart_clean.csv: {len(heart)} rows x {len(heart.columns)} columns")
    counts = heart["target"].value_counts()
    for label in ["yes", "no"]:
        share = 100 * counts[label] / len(heart)
        print(f"  target={label}: {counts[label]} ({share:.1f}%)")

    print("Mean values by target:")
    summary = heart.groupby("target")[["age", "chol", "thalach", "oldpeak"]].mean()
    for label in ["yes", "no"]:
        row = summary.loc[label]
        print(
            f"  {label}: age={row['age']:.1f}, chol={row['chol']:.1f}, "
            f"thalach={row['thalach']:.1f}, oldpeak={row['oldpeak']:.2f}"
        )

    best = comparison.loc[comparison["Accuracy"].idxmax()]
    print(
        f"Best model by accuracy: {best['Model']} "
        f"(accuracy={best['Accuracy']:.4f}, AUC={best['AUC']:.4f})"
    )


if __name__ == "__main__":
    main()
