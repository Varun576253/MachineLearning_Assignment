# ML Assignment 1 — Polynomial Regression

**Student:** BT2024146

**GitHub repository:** [MachineLearning_Assignment](https://github.com/Varun576253/MachineLearning_Assignment)

## 1. Overview

This project fits polynomial regression models for the two assigned problems, `var1` and `var2`. Each model predicts the continuous target `y` from its supplied features. The permitted polynomial degree ranges are 1–10 for `var1` and 1–20 for `var2`.

## 2. Final Model Results

Scores are mean validation metrics from the repeated cross-validation procedure described below.

| Problem | Model | Degree | Alpha | CV MSE | CV R² |
|---|---|---:|---:|---:|---:|
| var1 | Ridge Polynomial Regression | 5 | 20 | 0.515157 | 0.951535 |
| var2 | Ridge Polynomial Regression | 10 | 1 | 0.262251 | 0.994186 |

Degree 11 achieved the lowest raw cross-validation MSE for `var2` (0.260706). However, degree 10 was selected using the one-standard-error rule because its validation error was within one standard error of the minimum while requiring fewer polynomial terms (285 versus 363).

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

The solution expands each feature set into polynomial terms and compares polynomial Ridge regression across every permitted degree. Ordinary least-squares polynomial regression is also evaluated where the expanded feature count is below 800. Expanded features are standardized within each validation fold: the scaler is fitted only on that fold's training rows and then applied to its validation rows.

Model selection uses shuffled 5-fold cross-validation repeated twice (10 validation folds total, random seed 2026). Each fold uses 800 training rows and 200 validation rows. The selected degree is the lowest degree within one standard error of the minimum mean validation MSE; within that degree, the candidate with the lowest mean validation MSE is selected. The chosen pipeline is then refitted on all 1,000 training rows for each problem and used to predict its corresponding test rows.

## 6. Reproduction

To reproduce my results, place the five instructor-provided CSV files listed below in `data/`. I used Python 3.12. From the project root, run these commands in Windows PowerShell to create the environment, install the required packages, repeat model selection, and generate the prediction files. The input files are read without modification.

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
