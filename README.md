# ML Assignment 1 — Polynomial Regression

**Student:** BT2024146

**GitHub repository:** [MachineLearning_Assignment](https://github.com/Varun576253/MachineLearning_Assignment)

## 1. Overview

This project fits polynomial regression models for the two assigned problems, `var1` and `var2`. Each model predicts the continuous target `y` from its supplied features. The permitted polynomial degree ranges are 1–10 for `var1` and 1–20 for `var2`.

## 2. Final Model Results

Scores are mean validation metrics from the repeated cross-validation procedure described below.

| Problem | Model | Degree | Alpha | CV MSE | CV R² |
|---|---|---:|---:|---:|---:|
| var1 | Lasso Polynomial Regression | 5 | 0.007 | 0.333088 | 0.968708 |
| var2 | Ridge Polynomial Regression | 10 | 1 | 0.262251 | 0.994186 |

Degree 11 achieved the lowest raw cross-validation MSE for `var2` (0.260706). However, degree 10 was selected using the one-standard-error rule because its validation error was within one standard error of the minimum while requiring fewer polynomial terms (285 versus 363). For `var1`, degree 5 Lasso was selected as the global minimum CV MSE, outperforming Ridge by inducing sparsity across the 461 expanded polynomial terms.

### Regularization Rationale: Lasso vs. Ridge Selection

- **Why Lasso for `var1`:** The system involves 6 continuous operational parameters that expand combinatorially to 461 monomials at degree 5 (and up to 8,007 terms at degree 10). In multi-variable engineering systems, physical behavior is governed by a sparse subset of interaction terms. Ridge regression ($L_2$ penalty) retains all 461 terms, accumulating estimation variance from noise monomials (best Ridge CV MSE: 0.515157). In contrast, Lasso regression ($L_1$ penalty) drives non-informative terms to exactly zero, retaining only ~139 active features at $\alpha=0.007$. Eliminating over 70% of the noisy terms drops validation error by 35.3% to **0.333088** and boosts CV $R^2$ to **0.9687**.
- **Why Ridge for `var2`:** The task measures 3 spatial coordinate offsets $(x_1, x_2, x_3)$ representing physical sensor locations. The underlying thermal anomaly field is governed by continuous heat diffusion, where coordinates interact across dimensions in a smooth, dense polynomial manifold rather than isolated sparse terms. Expanding 3 coordinates produces 285 terms at degree 10, which exhibit severe geometric multicollinearity. Ridge's $L_2$ penalty provides uniform, stable shrinkage of the collinear eigenvalues without discarding continuous spatial interaction terms, achieving exceptional predictive accuracy (CV MSE: **0.262251**, CV $R^2$: **0.9942**).

## 3. Submission Deliverables

The first three rows are the primary assignment deliverables.

| Assignment deliverable | Location |
|---|---|
| Report PDF | `report/report.pdf` |
| var1 prediction file | `outputs/BT2024146_pred_var1.csv` |
| var2 prediction file | `outputs/BT2024146_pred_var2.csv` |
| Training and model-selection code | `src/model_selection.py` |
| Final training and inference code | `src/train_and_predict.py` |

## 4. Recommended Evaluation / Inspection Order

1. Open `report/report.pdf` for the methodology and final results.
2. Check `outputs/BT2024146_pred_var1.csv`.
3. Check `outputs/BT2024146_pred_var2.csv`.
4. Inspect `src/model_selection.py` for degree and model selection.
5. Inspect `src/train_and_predict.py` for final fitting and prediction generation.
6. Optionally inspect the model-selection CSV files in `outputs/`.

## 5. Methodology

The solution expands each feature set into polynomial terms and compares regularized polynomial regression (Ridge and Lasso) across every permitted degree. Ordinary least-squares polynomial regression is also evaluated where the expanded feature count is below 800. Expanded features are standardized within each validation fold: the scaler is fitted only on that fold's training rows and then applied to its validation rows.

Model selection uses shuffled 5-fold cross-validation repeated twice (10 validation folds total, random seed 2026). Each fold uses 800 training rows and 200 validation rows. The selected degree is the lowest degree within one standard error of the minimum mean validation MSE; within that degree, the candidate with the lowest mean validation MSE is selected. The chosen pipeline is then refitted on all 1,000 training rows for each problem and used to predict its corresponding test rows.

## 6. Reproduction

The five instructor-provided CSV files listed below are included in `data/`. I used Python 3.12. From the project root, run these commands in Windows PowerShell to create the environment, install the required packages, repeat model selection, and generate the prediction files. The input files are read without modification.

Required input files:

```text
data/BT2024146_train_var1.csv
data/BT2024146_test_var1.csv
data/BT2024146_train_var2.csv
data/BT2024146_test_var2.csv
data/sample_submission.csv
```

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/train_and_predict.py
```

The training script performs model selection, refits the final pipelines, and writes prediction and selection results to `outputs/`. The final report is included at `report/report.pdf`. To use a different input directory, pass `--data-dir PATH` to the training script.
