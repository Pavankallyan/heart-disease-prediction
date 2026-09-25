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
└── Presentation/  # project presentation (.pptx)
```

## How to run
Open `Code/heart analysis Final Project .R` in RStudio and run top to bottom.
