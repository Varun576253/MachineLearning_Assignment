"""Fold-safe polynomial model search for the two assignment datasets."""

from __future__ import annotations

from math import comb, sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


N_SPLITS = 5
N_REPEATS = 2
RANDOM_SEED = 2026
RIDGE_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0)


def polynomial_term_count(n_features: int, degree: int) -> int:
    """Number of non-constant monomials with total degree at most ``degree``."""
    return comb(n_features + degree, degree) - 1


def make_pipeline(family: str, degree: int, alpha: float | None = None) -> Pipeline:
    """Build the final polynomial-regression estimator."""
    steps: list[tuple[str, Any]] = [
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("scaler", StandardScaler()),
    ]
    if family == "Ridge":
        if alpha is None:
            raise ValueError("Ridge requires an alpha value.")
        steps.append(
            (
                "ridge",
                Ridge(
                    alpha=float(alpha),
                    fit_intercept=True,
                    solver="lsqr",
                    tol=1e-8,
                    max_iter=20_000,
                ),
            )
        )
    elif family == "OLS":
        steps.append(("ols", LinearRegression()))
    else:
        raise ValueError(f"Unknown model family: {family}")
    return Pipeline(steps)


def _mse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    residual = y_true - y_pred
    mse = float(np.mean(residual * residual))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual * residual) / total) if total else 0.0
    return mse, r2


def _new_candidate(family: str, degree: int, alpha: float | None, terms: int) -> dict[str, Any]:
    return {
        "family": family,
        "degree": degree,
        "alpha": alpha,
        "polynomial_terms": terms,
        "fold_validation_mse": [],
        "fold_validation_r2": [],
        "fold_training_mse": [],
        "fold_training_r2": [],
    }


def _finish_candidate(row: dict[str, Any], split_count: int) -> dict[str, Any]:
    val_mse = np.asarray(row.pop("fold_validation_mse"), dtype=float)
    val_r2 = np.asarray(row.pop("fold_validation_r2"), dtype=float)
    train_mse = np.asarray(row.pop("fold_training_mse"), dtype=float)
    train_r2 = np.asarray(row.pop("fold_training_r2"), dtype=float)
    row.update(
        {
            "mean_validation_mse": float(val_mse.mean()),
            "std_validation_mse": float(val_mse.std(ddof=1)),
            "se_validation_mse": float(val_mse.std(ddof=1) / sqrt(split_count)),
            "mean_validation_r2": float(val_r2.mean()),
            "std_validation_r2": float(val_r2.std(ddof=1)),
            "mean_training_mse": float(train_mse.mean()),
            "mean_training_r2": float(train_r2.mean()),
            "fold_validation_mse": ";".join(f"{x:.12g}" for x in val_mse),
            "fold_validation_r2": ";".join(f"{x:.12g}" for x in val_r2),
        }
    )
    return row


def _ridge_fold_metrics(
    train_design: np.ndarray,
    validation_design: np.ndarray,
    y_train: np.ndarray,
    y_validation: np.ndarray,
    alphas: tuple[float, ...],
) -> dict[float, tuple[tuple[float, float], tuple[float, float]]]:
    """Compute exact dense Ridge solutions for a fold in dual form.

    This reuses one eigendecomposition across the alpha grid. It is algebraically
    equivalent to Ridge on the standardized polynomial design, while avoiding
    redundant fits. The StandardScaler itself is fitted by the caller on the
    current fold's training rows only.
    """
    y_mean = float(y_train.mean())
    centered_y = y_train - y_mean
    gram_train = train_design @ train_design.T
    gram_validation = validation_design @ train_design.T
    eigenvalues, eigenvectors = eigh(gram_train, check_finite=False, driver="evd")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    projected_y = eigenvectors.T @ centered_y
    result: dict[float, tuple[tuple[float, float], tuple[float, float]]] = {}
    for alpha in alphas:
        dual_weights = eigenvectors @ (projected_y / (eigenvalues + alpha))
        train_prediction = y_mean + gram_train @ dual_weights
        validation_prediction = y_mean + gram_validation @ dual_weights
        result[alpha] = (
            _mse_r2(y_train, train_prediction),
            _mse_r2(y_validation, validation_prediction),
        )
    return result


def select_model(
    X: pd.DataFrame,
    y: pd.Series,
    max_degree: int,
    label: str = "problem",
    ridge_alphas: tuple[float, ...] = RIDGE_ALPHAS,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compare OLS and Ridge using repeated shuffled 5-fold CV.

    Polynomial expansion is deterministic (it learns no statistics) and is
    computed once per degree. Each fold gets its own StandardScaler fitted only
    on that fold's training design. Ridge uses the exact dual solution for
    efficiency; the final estimator is refit with sklearn's Ridge pipeline.

    The one-standard-error rule chooses the lowest polynomial degree whose mean
    validation MSE is within one fold-to-fold standard error of the best mean
    MSE. Within that degree, it chooses the candidate with the lowest mean MSE.
    """
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    n_features = X_array.shape[1]
    degrees = list(range(1, max_degree + 1))
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    splits = list(cv.split(X_array, y_array))
    ols_degrees = [
        degree
        for degree in degrees
        if polynomial_term_count(n_features, degree) < int(0.8 * len(X_array))
    ]
    rows: list[dict[str, Any]] = []

    for degree in degrees:
        polynomial = PolynomialFeatures(degree=degree, include_bias=False)
        design = polynomial.fit_transform(X_array)
        terms = design.shape[1]
        ridge_candidates = {
            alpha: _new_candidate("Ridge", degree, float(alpha), terms)
            for alpha in ridge_alphas
        }
        ols_candidate = (
            _new_candidate("OLS", degree, None, terms) if degree in ols_degrees else None
        )

        for train_index, validation_index in splits:
            scaler = StandardScaler()
            train_design = scaler.fit_transform(design[train_index])
            validation_design = scaler.transform(design[validation_index])
            y_train = y_array[train_index]
            y_validation = y_array[validation_index]

            ridge_scores = _ridge_fold_metrics(
                train_design,
                validation_design,
                y_train,
                y_validation,
                ridge_alphas,
            )
            for alpha, (train_score, validation_score) in ridge_scores.items():
                row = ridge_candidates[alpha]
                row["fold_training_mse"].append(train_score[0])
                row["fold_training_r2"].append(train_score[1])
                row["fold_validation_mse"].append(validation_score[0])
                row["fold_validation_r2"].append(validation_score[1])

            if ols_candidate is not None:
                coefficients, *_ = np.linalg.lstsq(train_design, y_train - y_train.mean(), rcond=None)
                train_prediction = y_train.mean() + train_design @ coefficients
                validation_prediction = y_train.mean() + validation_design @ coefficients
                train_score = _mse_r2(y_train, train_prediction)
                validation_score = _mse_r2(y_validation, validation_prediction)
                ols_candidate["fold_training_mse"].append(train_score[0])
                ols_candidate["fold_training_r2"].append(train_score[1])
                ols_candidate["fold_validation_mse"].append(validation_score[0])
                ols_candidate["fold_validation_r2"].append(validation_score[1])

        rows.extend(_finish_candidate(row, len(splits)) for row in ridge_candidates.values())
        if ols_candidate is not None:
            rows.append(_finish_candidate(ols_candidate, len(splits)))
        print(f"{label}: completed degree {degree}/{max_degree}", flush=True)

    results = pd.DataFrame(rows)
    best_index = int(results["mean_validation_mse"].idxmin())
    best = results.loc[best_index]
    cutoff = float(best["mean_validation_mse"] + best["se_validation_mse"])
    eligible = results[results["mean_validation_mse"] <= cutoff].copy()
    eligible["ridge_tiebreak"] = (eligible["family"] != "Ridge").astype(int)
    selected = eligible.sort_values(
        ["degree", "mean_validation_mse", "ridge_tiebreak", "alpha"],
        ascending=[True, True, True, False],
        na_position="last",
    ).iloc[0]
    record: dict[str, Any] = {
        "family": str(selected["family"]),
        "degree": int(selected["degree"]),
        "alpha": None if pd.isna(selected["alpha"]) else float(selected["alpha"]),
        "polynomial_terms": int(selected["polynomial_terms"]),
        "mean_validation_mse": float(selected["mean_validation_mse"]),
        "std_validation_mse": float(selected["std_validation_mse"]),
        "se_validation_mse": float(selected["se_validation_mse"]),
        "mean_validation_r2": float(selected["mean_validation_r2"]),
        "mean_training_mse": float(selected["mean_training_mse"]),
        "mean_training_r2": float(selected["mean_training_r2"]),
        "minimum_mean_validation_mse": float(best["mean_validation_mse"]),
        "one_se_cutoff_mse": cutoff,
        "selection_rule": "lowest degree within one fold-to-fold SE of minimum mean validation MSE; lowest MSE within that degree",
        "cv_n_splits": N_SPLITS,
        "cv_n_repeats": N_REPEATS,
        "cv_random_seed": RANDOM_SEED,
        "cv_shuffle": True,
        "ridge_alphas": list(ridge_alphas),
        "ols_degrees_compared": ols_degrees,
        "n_candidates": int(len(results)),
        "n_cv_folds": len(splits),
    }
    results["within_one_se_of_best"] = results["mean_validation_mse"] <= cutoff
    if record["alpha"] is None:
        alpha_match = results["alpha"].isna()
    else:
        alpha_match = np.isclose(results["alpha"], record["alpha"], equal_nan=False)
    results["selected"] = (
        (results["family"] == record["family"])
        & (results["degree"] == record["degree"])
        & alpha_match
    )
    results = results.sort_values(
        ["mean_validation_mse", "degree", "family", "alpha"],
        ascending=[True, True, True, True],
        na_position="first",
    ).reset_index(drop=True)
    return record, results


def evaluate_ridge_alphas(
    X: pd.DataFrame,
    y: pd.Series,
    degree: int,
    alphas: tuple[float, ...],
    label: str = "problem",
) -> pd.DataFrame:
    """Evaluate additional Ridge alphas on the same repeated-CV folds.

    Polynomial features are generated once. Each fold fits its scaler only on
    its training rows, matching ``select_model`` exactly. This is used for a
    small local alpha refinement after the full degree search.
    """
    if not alphas:
        raise ValueError("At least one Ridge alpha is required.")
    X_array = np.asarray(X, dtype=float)
    y_array = np.asarray(y, dtype=float)
    design = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(X_array)
    cv = RepeatedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_SEED,
    )
    splits = list(cv.split(X_array, y_array))
    candidates = {
        float(alpha): _new_candidate("Ridge", degree, float(alpha), design.shape[1])
        for alpha in alphas
    }

    for train_index, validation_index in splits:
        scaler = StandardScaler()
        train_design = scaler.fit_transform(design[train_index])
        validation_design = scaler.transform(design[validation_index])
        y_train = y_array[train_index]
        y_validation = y_array[validation_index]
        scores = _ridge_fold_metrics(
            train_design,
            validation_design,
            y_train,
            y_validation,
            tuple(candidates),
        )
        for alpha, (train_score, validation_score) in scores.items():
            row = candidates[alpha]
            row["fold_training_mse"].append(train_score[0])
            row["fold_training_r2"].append(train_score[1])
            row["fold_validation_mse"].append(validation_score[0])
            row["fold_validation_r2"].append(validation_score[1])

    results = pd.DataFrame(
        [_finish_candidate(row, len(splits)) for row in candidates.values()]
    )
    print(f"{label}: completed focused alpha refinement at degree {degree}", flush=True)
    return results.sort_values("alpha").reset_index(drop=True)
