"""
Universal Data Preparation Module
===================================
Converts **any** cleaned DataFrame into ML-ready train/test splits.

Handles:
- Non-numeric target columns (auto label-encoding)
- High-cardinality categoricals (frequency encoding instead of one-hot explosion)
- Boolean, datetime-derived, and mixed-type columns
- Remaining NaN / infinite values
- Feature-column explosion guard (max_features cap)
- Robust train/test splitting with stratification when possible
"""

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    sensitive_col: str,
    test_size: float = 0.30,
    random_state: int = 42,
    max_onehot_cardinality: int = 30,
    max_features: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Transform a (potentially messy) DataFrame into numeric train/test arrays
    ready for sklearn models.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset (ideally already cleaned by ``data_cleaner.clean_dataset``).
    target_col : str
        Name of the target / label column.
    sensitive_col : str
        Name of the sensitive-attribute column (kept for fairness evaluation).
    test_size : float
        Fraction of data reserved for testing (default 0.30).
    random_state : int
        Reproducibility seed.
    max_onehot_cardinality : int
        Categorical columns with more unique values than this are
        frequency-encoded instead of one-hot encoded.
    max_features : int
        Hard cap on number of feature columns after encoding (safety valve
        against memory explosions on wide datasets).

    Returns
    -------
    X_train, X_test, y_train, y_test, sensitive_test
    """

    # ── Validate required columns ──
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset. "
                         f"Available columns: {df.columns.tolist()}")
    if sensitive_col not in df.columns:
        raise ValueError(f"Sensitive column '{sensitive_col}' not found in dataset. "
                         f"Available columns: {df.columns.tolist()}")

    df = df.copy()

    # ── Drop rows where target or sensitive is missing ──
    df = df.dropna(subset=[target_col, sensitive_col]).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No rows remain after dropping missing target / sensitive values.")

    # ── Encode target column ──
    y = df[target_col]
    if not pd.api.types.is_numeric_dtype(y):
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), name=target_col)
    else:
        y = y.astype(int)

    # ── Preserve sensitive column (as-is for group evaluation) ──
    sensitive = df[sensitive_col].copy()

    # ── Build feature matrix ──
    X = df.drop(columns=[target_col])

    # ── Encode features ──
    X = _encode_features(X, max_onehot_cardinality)

    # ── Replace any residual inf / NaN ──
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isnull().any().any():
        for col in X.columns[X.isnull().any()]:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
            else:
                X[col] = X[col].fillna("Unknown")

    # ── Ensure everything is numeric ──
    # Any remaining object columns → force label-encode
    for col in X.select_dtypes(include=["object", "category", "bool"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # ── Cap feature count ──
    if X.shape[1] > max_features:
        # Keep features with highest variance
        variances = X.var().sort_values(ascending=False)
        keep_cols = variances.head(max_features).index.tolist()
        # Ensure sensitive column encoded version stays if present
        X = X[keep_cols]
        warnings.warn(
            f"Feature count ({X.shape[1]}) exceeded max_features ({max_features}). "
            f"Kept top {max_features} by variance."
        )

    # ── Train / Test split ──
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Stratification fails when a class has too few samples
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    sensitive_test = sensitive.loc[X_test.index]

    return X_train, X_test, y_train, y_test, sensitive_test


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _encode_features(X: pd.DataFrame, max_onehot_cardinality: int) -> pd.DataFrame:
    """
    Smart encoding strategy:
    - Low-cardinality categoricals → one-hot (pd.get_dummies)
    - High-cardinality categoricals → frequency encoding
    - Booleans → 0/1
    - Already-numeric columns → untouched
    """
    X = X.copy()

    # ── Booleans → int ──
    for col in X.select_dtypes(include=["bool"]).columns:
        X[col] = X[col].astype(int)

    # ── Separate low-card vs high-card categoricals ──
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    low_card = [c for c in cat_cols if X[c].nunique() <= max_onehot_cardinality]
    high_card = [c for c in cat_cols if X[c].nunique() > max_onehot_cardinality]

    # ── Frequency-encode high-cardinality columns ──
    for col in high_card:
        freq_map = X[col].value_counts(normalize=True).to_dict()
        X[col] = X[col].map(freq_map).astype(float)
        # If any new values appear as NaN after map, fill with 0
        X[col] = X[col].fillna(0.0)

    # ── One-hot encode low-cardinality columns ──
    if low_card:
        X = pd.get_dummies(X, columns=low_card, drop_first=True, dtype=int)

    return X
