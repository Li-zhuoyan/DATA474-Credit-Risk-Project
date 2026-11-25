# Credit Risk Prediction and High-Risk Group Profiling

A two-stage machine learning framework for loan default prediction and high-risk borrower segmentation, built as the final project for **DATA 474** at the University of Arizona.

## Overview

Lenders face a constant trade-off between approving risky loans and rejecting creditworthy applicants. A single default score says little about *why* a borrower is risky or what a bank should do about it. This project addresses both questions:

1. **Default prediction** — LASSO variable selection, ROSE oversampling, and two ensemble models (Random Forest and XGBoost).
2. **High-risk profiling** — PCA and K-means clustering to uncover distinct borrower personas and inform targeted credit strategies.

## Data

- Source: Kaggle credit risk dataset
- 32,581 borrower records, 12 variables
- Target: `loan_status` (0 = non-default, 1 = default)

## Methods

- **Cleaning:** logical filters (age ≥ 18, income ≥ 0, loan amount ≥ 0), mean/mode imputation, IQR-based outlier removal.
- **Encoding:** categorical features converted to factors and one-hot encoded.
- **Variable selection:** LASSO regression via `cv.glmnet` (alpha = 1, family = binomial); 17 predictors retained, `person_home_ownership_mortgage` dropped.
- **Balancing:** ROSE oversampling (p = 0.5) on the training set to address severe class imbalance.
- **Split:** 80% training / 20% testing.
- **Models:** Random Forest (100 trees) and XGBoost (100 rounds), evaluated by confusion matrix, ROC curve, and AUC.
- **Profiling:** Top 20% highest-risk borrowers → PCA (10 components, 84.9% variance explained) → K-means (k = 5, peak silhouette 0.249).

## Key results

- AUC: Random Forest **0.8921**, XGBoost **0.8884** — nearly identical, so model choice rested on practical factors (training speed, resource use, deployment ease).
- Actual default rate among the top-20% high-risk group: **~0.72**, over three times the overall rate of 0.218.
- Five distinct borrower profiles with tailored strategies — from short-term small loans for young, low-income borrowers to commercial credit for stable mid-career clients.

## Repository contents

- `Machine Learning for Credit Risk(2).pdf` — final report
- `end code.Rmd` — final R code
- `feature_importance.csv` — variable importance output
- `cluster_*.png`, `cluster_bar_chart_grouped.png`, `cluster_comparison_facet.png` — visualization figures
- `Machine Learning Project Proposal.docx` — project proposal
- `演示文稿1.pptx` — presentation slides

## How to reproduce

Open `end code.Rmd` in RStudio and run all code chunks sequentially.

## Team

Zhuoyan Li, Zhiyao Liu, Yiqi Hu — Department of Statistics and Data Science, The University of Arizona.
