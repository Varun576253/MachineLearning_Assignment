"""Select polynomial models by CV, refit on all training rows, and predict."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from model_selection import (
    RANDOM_SEED,
    RIDGE_ALPHAS,
    evaluate_ridge_alphas,
    make_pipeline,
    select_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROBLEMS = {"var1": 10, "var2": 20}
FOCUSED_ALPHAS = {
    "var1": (1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0),
    "var2": (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0),
}
MIN_RELATIVE_MSE_IMPROVEMENT = 0.01


def _fold_values(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(";")], dtype=float)


def _refine_alpha(
    X: pd.DataFrame,
    y: pd.Series,
    variable: str,
    selected: dict[str, Any],
    cv_results: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Compare a focused alpha set on the existing degree and CV procedure."""
    if selected["family"] != "Ridge" or selected["alpha"] is None:
        raise ValueError(f"Expected the established Ridge selection for {variable}.")

    degree = int(selected["degree"])
    baseline_alpha = float(selected["alpha"])
    focused = FOCUSED_ALPHAS[variable]
    known_alphas = tuple(float(alpha) for alpha in selected["ridge_alphas"])
    missing_alphas = tuple(alpha for alpha in focused if alpha not in known_alphas)
    new_results = (
        evaluate_ridge_alphas(X, y, degree, missing_alphas, label=variable)
        if missing_alphas
        else pd.DataFrame()
    )
    base = cv_results[
        (cv_results["family"] == "Ridge")
        & (cv_results["degree"] == degree)
        & np.isclose(cv_results["alpha"], baseline_alpha)
    ].iloc[0]
    base_folds = _fold_values(base["fold_validation_mse"])

    candidate_records: list[dict[str, Any]] = []
    for alpha in focused:
        if alpha in known_alphas:
            row = cv_results[
                (cv_results["family"] == "Ridge")
                & (cv_results["degree"] == degree)
                & np.isclose(cv_results["alpha"], alpha)
            ].iloc[0]
        else:
            row = new_results[np.isclose(new_results["alpha"], alpha)].iloc[0]
        folds = _fold_values(row["fold_validation_mse"])
        improvement_by_fold = base_folds - folds
        paired_mean = float(improvement_by_fold.mean())
        paired_se = float(improvement_by_fold.std(ddof=1) / np.sqrt(len(improvement_by_fold)))
        relative = float(
            (float(base["mean_validation_mse"]) - float(row["mean_validation_mse"]))
            / float(base["mean_validation_mse"])
        )
        candidate_records.append(
            {
                "alpha": float(alpha),
                "degree": degree,
                "mean_validation_mse": float(row["mean_validation_mse"]),
                "mean_validation_r2": float(row["mean_validation_r2"]),
                "se_validation_mse": float(row["se_validation_mse"]),
                "relative_mse_improvement_vs_baseline": relative,
                "paired_mean_mse_improvement_vs_baseline": paired_mean,
                "paired_se_mse_improvement": paired_se,
                "fold_validation_mse": row["fold_validation_mse"],
                "is_baseline_alpha": bool(np.isclose(alpha, baseline_alpha)),
            }
        )
    refinement = pd.DataFrame(candidate_records).sort_values("alpha").reset_index(drop=True)
    best_local = refinement.sort_values("mean_validation_mse").iloc[0]
    relative_improvement = float(best_local["relative_mse_improvement_vs_baseline"])
    paired_improvement = float(best_local["paired_mean_mse_improvement_vs_baseline"])
    paired_se = float(best_local["paired_se_mse_improvement"])
    meaningful = bool(
        not np.isclose(float(best_local["alpha"]), baseline_alpha)
        and relative_improvement >= MIN_RELATIVE_MSE_IMPROVEMENT
        and paired_improvement > paired_se
    )

    rerun = False
    final_selected = selected
    final_results = cv_results
    if meaningful:
        expanded_alphas = tuple(sorted(set(RIDGE_ALPHAS).union(focused)))
        final_selected, final_results = select_model(
            X,
            y,
            max_degree=PROBLEMS[variable],
            label=f"{variable} refined full search",
            ridge_alphas=expanded_alphas,
        )
        rerun = True

    refinement["final_selected"] = (
        (refinement["degree"] == int(final_selected["degree"]))
        & np.isclose(refinement["alpha"], float(final_selected["alpha"]))
    )
    refinement.to_csv(
        output_dir / f"alpha_refinement_{variable}.csv",
        index=False,
        float_format="%.12g",
    )
    record = {
        "baseline_degree": degree,
        "baseline_alpha": baseline_alpha,
        "baseline_mean_validation_mse": float(base["mean_validation_mse"]),
        "focused_alphas": list(focused),
        "best_local_alpha": float(best_local["alpha"]),
        "best_local_mean_validation_mse": float(best_local["mean_validation_mse"]),
        "best_local_mean_validation_r2": float(best_local["mean_validation_r2"]),
        "relative_mse_improvement": relative_improvement,
        "paired_mean_mse_improvement": paired_improvement,
        "paired_se_mse_improvement": paired_se,
        "minimum_relative_mse_improvement": MIN_RELATIVE_MSE_IMPROVEMENT,
        "meaningful_improvement": meaningful,
        "decision": "accepted; reran full degree search" if meaningful else "retained baseline",
        "full_search_rerun": rerun,
        "final_degree": int(final_selected["degree"]),
        "final_alpha": float(final_selected["alpha"]),
    }
    return final_selected, final_results, record


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def diagnose(train: pd.DataFrame, test: pd.DataFrame, variable: str) -> dict[str, Any]:
    feature_columns = [column for column in train.columns if column != "y"]
    return {
        "variable": variable,
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_columns": train.columns.tolist(),
        "test_columns": test.columns.tolist(),
        "feature_columns": feature_columns,
        "train_schema_matches_test_features": feature_columns == test.columns.tolist(),
        "train_missing_by_column": train.isna().sum().to_dict(),
        "test_missing_by_column": test.isna().sum().to_dict(),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_feature_rows": int(test.duplicated().sum()),
        "feature_ranges_train": {
            column: [float(train[column].min()), float(train[column].max())]
            for column in feature_columns
        },
        "feature_ranges_test": {
            column: [float(test[column].min()), float(test[column].max())]
            for column in feature_columns
        },
        "target_summary": {
            "min": float(train["y"].min()),
            "q01": float(train["y"].quantile(0.01)),
            "q25": float(train["y"].quantile(0.25)),
            "median": float(train["y"].median()),
            "mean": float(train["y"].mean()),
            "std": float(train["y"].std(ddof=1)),
            "q75": float(train["y"].quantile(0.75)),
            "q99": float(train["y"].quantile(0.99)),
            "max": float(train["y"].max()),
        },
    }


def _check_inputs(train: pd.DataFrame, test: pd.DataFrame, variable: str) -> list[str]:
    if "y" not in train.columns:
        raise ValueError(f"Training file for {variable} is missing target column 'y'.")
    feature_columns = [column for column in train.columns if column != "y"]
    if test.columns.tolist() != feature_columns:
        raise ValueError(
            f"Train/test feature schema mismatch for {variable}: "
            f"{feature_columns} vs {test.columns.tolist()}"
        )
    if train.empty or test.empty:
        raise ValueError(f"Empty data supplied for {variable}.")
    if train.isna().any().any() or test.isna().any().any():
        raise ValueError(f"Missing values supplied for {variable}.")
    if not np.isfinite(train.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite training values supplied for {variable}.")
    if not np.isfinite(test.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite test values supplied for {variable}.")
    return feature_columns


def run(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    if sample.columns.tolist() != ["y"] or len(sample) != 1000:
        raise ValueError("sample_submission.csv must contain 1000 rows in one 'y' column.")

    report: dict[str, Any] = {
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "validation": {
            "strategy": "RepeatedKFold",
            "n_splits": 5,
            "n_repeats": 2,
            "n_validation_folds": 10,
            "train_fraction_per_fold": 0.8,
            "validation_fraction_per_fold": 0.2,
            "shuffle": True,
            "random_seed": RANDOM_SEED,
            "selection_rule": "lowest degree within one fold-to-fold SE of minimum mean validation MSE",
        },
        "sample_submission_rows": int(len(sample)),
        "problems": {},
    }
    diagnostic_records: list[dict[str, Any]] = []

    for variable, max_degree in PROBLEMS.items():
        train_path = data_dir / f"BT2024146_train_{variable}.csv"
        test_path = data_dir / f"BT2024146_test_{variable}.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        feature_columns = _check_inputs(train, test, variable)
        if len(test) != len(sample):
            raise ValueError(
                f"{variable} test rows ({len(test)}) do not match the sample submission ({len(sample)})."
            )

        diagnostics = diagnose(train, test, variable)
        if not diagnostics["train_schema_matches_test_features"]:
            raise ValueError(f"Train/test schema mismatch for {variable}.")
        diagnostic_records.append(diagnostics)

        X_train = train[feature_columns]
        y_train = train["y"]
        X_test = test[feature_columns]
        selected, cv_results = select_model(
            X_train,
            y_train,
            max_degree=max_degree,
            label=variable,
        )
        selected, cv_results, alpha_refinement = _refine_alpha(
            X_train,
            y_train,
            variable,
            selected,
            cv_results,
            output_dir,
        )
        cv_path = output_dir / f"model_selection_{variable}.csv"
        cv_results.to_csv(cv_path, index=False, float_format="%.12g")

        final_model = make_pipeline(selected["family"], selected["degree"], selected["alpha"])
        final_model.fit(X_train, y_train)
        predictions = np.asarray(final_model.predict(X_test), dtype=float)
        if predictions.shape != (len(test),):
            raise ValueError(f"Unexpected prediction shape for {variable}: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise ValueError(f"Non-finite predictions generated for {variable}.")

        filename = f"BT2024146_pred_{variable}.csv"
        prediction_path = output_dir / filename
        pd.DataFrame({"y": predictions}).to_csv(
            prediction_path,
            index=False,
            float_format="%.12f",
        )
        written = pd.read_csv(prediction_path)
        if written.columns.tolist() != ["y"] or len(written) != 1000:
            raise ValueError(f"Invalid prediction CSV schema or row count: {prediction_path}")
        written_predictions = written["y"].to_numpy(dtype=float)
        if not np.isfinite(written_predictions).all():
            raise ValueError(f"Invalid prediction values written to {prediction_path}")
        if not np.allclose(written_predictions, predictions, rtol=0.0, atol=5e-13):
            raise ValueError(f"Written predictions changed values or row order: {prediction_path}")

        report["problems"][variable] = {
            "max_allowed_degree": max_degree,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_features": int(len(feature_columns)),
            "features": feature_columns,
            "selected": selected,
            "alpha_refinement": alpha_refinement,
            "cv_results_file": cv_path.name,
            "prediction_file": filename,
            "prediction_min": float(predictions.min()),
            "prediction_max": float(predictions.max()),
            "prediction_mean": float(predictions.mean()),
            "prediction_std": float(predictions.std(ddof=1)),
            "prediction_rows": int(len(predictions)),
            "prediction_column": "y",
        }
        print(
            f"{variable}: {selected['family']} degree={selected['degree']} "
            f"alpha={selected['alpha']} MSE={selected['mean_validation_mse']:.8g} "
            f"R2={selected['mean_validation_r2']:.8g} -> {prediction_path}"
        )

    (output_dir / "data_diagnostics.json").write_text(
        json.dumps(diagnostic_records, indent=2, default=_json_value), encoding="utf-8"
    )
    summary_path = output_dir / "selection_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, default=_json_value), encoding="utf-8")
    print(f"Saved model selection summary to {summary_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.data_dir.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
