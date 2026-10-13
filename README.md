# Heart Disease Risk Prediction

Heart disease risk classification in R — model comparison study.

## Overview
Binary classification of heart disease risk using clinical features, comparing
Logistic Regression and K-Nearest Neighbors.

## Approach
- Data cleaning and exploratory analysis (`Images/` — distributions by target,
  boxplots, histograms, pairplots)
- Model training: Logistic Regression vs KNN (`Code/heart analysis Final Project .R`)
- ROC curve analysis for both models (`Images/roc_logistic.png`, `Images/roc_knn.png`)
- Model comparison summary (`Data/model_comparison.csv`)

## Project structure
```
├── Code/          # analysis script (.R)
├── Data/          # heart_clean.csv, model_comparison.csv
├── Images/        # EDA plots and ROC curves
├── Presentation/  # project presentation (.pptx)
├── tests/         # pytest data-validation suite
└── requirements.txt
```

## Results

From `Data/model_comparison.csv` (test-set metrics):

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic Regression | 0.8431 | 0.8857 | 0.7980 | 0.9286 |
| KNN (k=3) | 0.8775 | 0.8381 | 0.9192 | 0.9859 |

KNN wins on accuracy and AUC; logistic regression catches slightly more true
positives. ROC curves live in `Images/roc_logistic.png` and `Images/roc_knn.png`.

## Data dictionary

`Data/heart_clean.csv` — 1025 rows, 14 columns, no missing values.

| Column | Meaning |
|---|---|
| `age` | Age in years |
| `sex` | Sex (1 = male, 0 = female) |
| `cp` | Chest pain type (1–4) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia type (3 = normal, 6 = fixed, 7 = reversible) |
| `target` | Heart disease present (`yes` / `no`) |

## Reproducibility

- R packages: `tidyverse`, `caret`, `pROC`, `GGally`
- Random seed: `set.seed(123)`; 80/20 stratified split via `caret::createDataPartition`
- KNN tuned over k = 3–31 (step 2); best k = 3
- The script expects the raw dataset as `heart.csv` in the same folder as the
  `.R` file; it writes `heart_clean.csv` and `model_comparison.csv`

## Data checks

`tests/test_data.py` validates the committed data artifacts (schema, no
missing values, binary target, sane clinical ranges, valid model metrics).
Install deps with `pip install -r requirements.txt` and run `pytest tests/`.

## How to run
Open `Code/heart analysis Final Project .R` in RStudio and run top to bottom.
