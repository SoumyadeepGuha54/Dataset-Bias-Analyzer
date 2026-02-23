import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif


def cramers_v(x, y):
    """Compute Cramér's V statistic for two categorical Series."""
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.size == 0:
        return 0.0
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    denominator = n * (min(r, k) - 1)
    if denominator == 0:
        return 0.0
    return np.sqrt(chi2 / denominator)


def detect_proxies(df, sensitive_col, threshold=0.7):
    """
    Detect potential proxy variables for a sensitive attribute.

    For each feature in the DataFrame (excluding the sensitive column itself):
      - Categorical features: Cramér's V
      - Numerical features: Mutual Information (normalized to [0, 1])

    Parameters
    ----------
    df : pd.DataFrame
        The full DataFrame (before one-hot encoding).
    sensitive_col : str
        Name of the sensitive attribute column.
    threshold : float
        Correlation threshold above which a feature is flagged.

    Returns
    -------
    dict
        {feature_name: correlation_score} for every flagged proxy variable.
    """
    if sensitive_col not in df.columns:
        return {}

    sensitive = df[sensitive_col]
    feature_cols = [c for c in df.columns if c != sensitive_col]

    proxy_flags = {}

    # --- Categorical features → Cramér's V ---
    cat_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        score = cramers_v(df[col].astype(str), sensitive.astype(str))
        if score > threshold:
            proxy_flags[col] = round(float(score), 4)

    # --- Numerical features → Mutual Information ---
    num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        # Encode sensitive as integer labels for MI computation
        sensitive_encoded = pd.Categorical(sensitive).codes
        X_num = df[num_cols].fillna(0)

        mi_scores = mutual_info_classif(
            X_num, sensitive_encoded, discrete_features=False, random_state=42
        )

        # Normalize MI scores to [0, 1] range
        max_mi = mi_scores.max() if mi_scores.max() > 0 else 1.0
        for col, mi in zip(num_cols, mi_scores):
            normalized = mi / max_mi
            if normalized > threshold:
                proxy_flags[col] = round(float(normalized), 4)

    return proxy_flags
