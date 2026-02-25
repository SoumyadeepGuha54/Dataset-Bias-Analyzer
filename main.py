"""
Enhanced Bias Analysis Pipeline
================================
Orchestrates data quality analysis, universal cleaning, multi-model training,
and fairness evaluation.  Works with **any** tabular CSV dataset.
"""

import traceback
import warnings

import numpy as np
import pandas as pd

from engine.data_processor import prepare_data
from engine.model_trainer import (
    train_baseline,
    train_multiple_models,
    get_feature_importance,
)
from engine.fairness_metrics import compute_group_metrics, compute_fairness_scores
from engine.bias_report import generate_bias_flag
from engine.proxy_detector import detect_proxies
from engine.data_cleaner import analyze_data_quality, clean_dataset


def run_bias_engine(
    df,
    target_col,
    sensitive_col,
    models=None,
    cleaning_config=None,
    auto_clean=True,
):
    """
    Universal bias-analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset (any shape / types / messiness).
    target_col : str
        Target / label column name.
    sensitive_col : str
        Sensitive-attribute column name.
    models : list[str] | None
        Model keys to train (``None`` → legacy single Logistic Regression).
    cleaning_config : dict | None
        Explicit cleaning flags passed to ``clean_dataset()``.  If *None*
        and ``auto_clean=True``, a sensible default config is used so that
        every dataset is guaranteed to be processable.
    auto_clean : bool
        When True **and** ``cleaning_config is None``, automatically apply
        safe cleaning (duplicates, missing, infinite, whitespace, ID/constant
        column removal, high-cardinality encoding, mixed-type fixing,
        datetime handling).  Does NOT scale or SMOTE by default.

    Returns
    -------
    dict
        Comprehensive results dict (see docstring body for keys).
    """

    # ── Initialise result container ──
    results = {
        "models": {},
        "proxy_variables": {},
        "cleaning_report": None,
        "data_quality": None,
    }

    # ── Step 1 – Data Quality Snapshot ──
    try:
        results["data_quality"] = analyze_data_quality(df, target_col)
    except Exception as exc:
        warnings.warn(f"Quality analysis encountered an issue: {exc}")
        results["data_quality"] = {}

    # ── Step 2 – Cleaning ──
    df_processed = df.copy()

    if cleaning_config is not None:
        # Explicit user config
        df_processed, cleaning_report = clean_dataset(
            df_processed,
            target_col=target_col,
            sensitive_col=sensitive_col,
            **cleaning_config,
        )
        results["cleaning_report"] = cleaning_report
    elif auto_clean:
        # Default safe-clean so ANY dataset can proceed
        df_processed, cleaning_report = clean_dataset(
            df_processed,
            target_col=target_col,
            sensitive_col=sensitive_col,
            remove_duplicates=True,
            impute_missing=True,
            handle_outliers=True,
            scale_features=False,
            balance_classes=False,
            drop_id_columns=True,
            drop_constant_columns=True,
            fix_whitespace=True,
            handle_infinite=True,
            handle_datetime=True,
            handle_mixed_types=True,
            high_cardinality_method="frequency",
        )
        results["cleaning_report"] = cleaning_report

    # ── Step 3 – Proxy Detection (before encoding) ──
    try:
        results["proxy_variables"] = detect_proxies(
            df_processed.drop(columns=[target_col], errors="ignore"),
            sensitive_col,
        )
    except Exception as exc:
        warnings.warn(f"Proxy detection skipped: {exc}")
        results["proxy_variables"] = {}

    # ── Step 4 – Data Preparation ──
    X_train, X_test, y_train, y_test, sensitive_test = prepare_data(
        df_processed, target_col, sensitive_col
    )

    # Store feature names (after encoding)
    feature_names = (
        X_train.columns.tolist()
        if hasattr(X_train, "columns")
        else [f"feature_{i}" for i in range(X_train.shape[1])]
    )
    results["feature_names"] = feature_names

    # ── Step 5 – Model Training ──
    if models is None or len(models) == 0:
        # Legacy path – single Logistic Regression
        model, y_pred, accuracy = train_baseline(X_train, y_train, X_test, y_test)

        group_metrics = compute_group_metrics(y_test, y_pred, sensitive_test)
        fairness_scores = compute_fairness_scores(group_metrics)
        bias_flag = generate_bias_flag(fairness_scores)

        return {
            "accuracy": accuracy,
            "group_metrics": group_metrics,
            "fairness_scores": fairness_scores,
            "bias_detected": bias_flag,
            "proxy_variables": results["proxy_variables"],
        }

    # Multi-model training
    model_results = train_multiple_models(X_train, y_train, X_test, y_test, models)

    # ── Step 6 – Fairness Evaluation per model ──
    for model_name, model_data in model_results.items():
        y_pred = model_data["predictions"]

        group_metrics = compute_group_metrics(y_test, y_pred, sensitive_test)
        fairness_scores = compute_fairness_scores(group_metrics)
        bias_flag = generate_bias_flag(fairness_scores)

        feature_importance = get_feature_importance(
            model_data["model"], feature_names, model_name
        )

        results["models"][model_name] = {
            "predictions": y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred,
            "probabilities": (
                model_data["probabilities"].tolist()
                if model_data["probabilities"] is not None
                else None
            ),
            "metrics": model_data["metrics"],
            "group_metrics": group_metrics,
            "fairness_scores": fairness_scores,
            "bias_detected": bias_flag,
            "feature_importance": feature_importance,
        }

    # Test data for visualisations
    results["test_data"] = {
        "y_test": y_test.tolist() if hasattr(y_test, "tolist") else y_test,
        "sensitive_test": (
            sensitive_test.tolist()
            if hasattr(sensitive_test, "tolist")
            else sensitive_test
        ),
    }

    return results


# ── Legacy shim ──
def run_bias_engine_legacy(df, target_col, sensitive_col):
    """Backward-compatible wrapper (single Logistic Regression, no cleaning)."""
    return run_bias_engine(
        df, target_col, sensitive_col, models=None, cleaning_config=None, auto_clean=True
    )


if __name__ == "__main__":
    pass
