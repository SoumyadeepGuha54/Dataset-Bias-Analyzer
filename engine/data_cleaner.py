"""
Universal Data Quality Analysis & Cleaning Engine
===================================================
Production-grade module that handles **any** CSV dataset regardless of shape,
types, encoding quirks, or messiness.

Capabilities
------------
- Detect & report every common quality issue (missing values, duplicates,
  outliers, infinite values, constant / near-constant columns, high-cardinality
  categoricals, mixed-type columns, whitespace pollution, date-time columns,
  ID-like columns, class imbalance).
- Clean datasets with granular, configurable options.
- Generate before/after cleaning reports with full statistics.
"""

import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# SMOTE import – gracefully unavailable
try:
    from imblearn.over_sampling import SMOTE, SMOTENC
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────

def _is_numeric_dtype(series: pd.Series) -> bool:
    """Return True for any numeric dtype (int/float, any width, nullable)."""
    return pd.api.types.is_numeric_dtype(series)


def _is_categorical_dtype(series: pd.Series) -> bool:
    """Return True for object, string, category, or boolean dtypes."""
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_categorical_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_string_dtype(series)
    )


def _is_datetime_dtype(series: pd.Series) -> bool:
    """Return True for datetime64 or timedelta64 dtypes."""
    return pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series)


def _looks_like_id(series: pd.Series, name: str) -> bool:
    """Heuristic: column is likely an ID / row-index if it's unique and
    monotonic or its name matches common ID patterns."""
    name_lower = name.lower().strip()
    id_patterns = [
        r"^id$", r"^_id$", r"_id$", r"^index$", r"^row_?num",
        r"^unnamed", r"^serial", r"^record_?id", r"^uid$",
    ]
    if any(re.search(p, name_lower) for p in id_patterns):
        return True
    # Unique monotonic integer → very likely ID
    if _is_numeric_dtype(series) and series.is_unique and series.is_monotonic_increasing:
        if len(series) > 20:
            return True
    return False


def _looks_like_datetime(series: pd.Series) -> bool:
    """Try to infer if a non-datetime object column actually contains date strings."""
    if _is_datetime_dtype(series):
        return True
    if not pd.api.types.is_object_dtype(series):
        return False
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(sample, errors="coerce")
        success_rate = converted.notna().sum() / len(sample)
        return success_rate > 0.8
    except Exception:
        return False


def _detect_high_cardinality(series: pd.Series, threshold: float = 0.90) -> bool:
    """Flag a categorical column if >90 % of values are unique (likely free text / IDs)."""
    if _is_numeric_dtype(series):
        return False
    n_unique = series.nunique(dropna=True)
    n_total = series.count()
    if n_total == 0:
        return False
    return (n_unique / n_total) > threshold


def _detect_constant_column(series: pd.Series) -> bool:
    """Return True if column has only a single unique value (or all NaN)."""
    return series.nunique(dropna=True) <= 1


def _detect_near_constant_column(series: pd.Series, threshold: float = 0.99) -> bool:
    """Return True if one value dominates ≥ threshold of non-null entries."""
    if series.count() == 0:
        return True
    top_freq = series.value_counts(normalize=True, dropna=True).iloc[0]
    return top_freq >= threshold


# ─────────────────────────────────────────────────────────────
# Quality Analysis
# ─────────────────────────────────────────────────────────────

def analyze_data_quality(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """
    Comprehensive dataset quality audit.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    target_col : str, optional
        Target column name for class-imbalance analysis.

    Returns
    -------
    dict
        {
            total_rows, total_columns,
            missing_values       – {col: count},
            missing_pct          – {col: pct},
            infinite_values      – {col: count},
            duplicates           – int,
            outliers             – {col: count},
            dtypes               – {col: dtype_str},
            constant_columns     – [col, ...],
            near_constant_columns – [col, ...],
            id_columns           – [col, ...],
            high_cardinality     – [col, ...],
            datetime_columns     – [col, ...],
            mixed_type_columns   – [col, ...],
            whitespace_issues    – {col: count},
            class_distribution   – {class: count} | None,
        }
    """
    report: Dict[str, Any] = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "missing_pct": {},
        "infinite_values": {},
        "duplicates": 0,
        "outliers": {},
        "dtypes": {},
        "constant_columns": [],
        "near_constant_columns": [],
        "id_columns": [],
        "high_cardinality": [],
        "datetime_columns": [],
        "mixed_type_columns": [],
        "whitespace_issues": {},
        "class_distribution": None,
    }

    n_rows = len(df)

    # ── Missing values ──
    missing = df.isnull().sum()
    for col, cnt in missing.items():
        if cnt > 0:
            report["missing_values"][col] = int(cnt)
            report["missing_pct"][col] = round(cnt / n_rows * 100, 2)

    # ── Infinite values (numeric only) ──
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            report["infinite_values"][col] = int(inf_count)

    # ── Duplicates ──
    report["duplicates"] = int(df.duplicated().sum())

    # ── Data types ──
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # ── Column-level diagnostics ──
    for col in df.columns:
        series = df[col]

        # Constant / near-constant
        if _detect_constant_column(series):
            report["constant_columns"].append(col)
        elif _detect_near_constant_column(series):
            report["near_constant_columns"].append(col)

        # ID-like
        if _looks_like_id(series, col):
            report["id_columns"].append(col)

        # High cardinality
        if _detect_high_cardinality(series):
            report["high_cardinality"].append(col)

        # Datetime (explicit or latent)
        if _looks_like_datetime(series):
            report["datetime_columns"].append(col)

        # Mixed types (object column containing both numbers and strings)
        if pd.api.types.is_object_dtype(series):
            sample = series.dropna().head(200)
            if len(sample) > 0:
                numeric_mask = sample.apply(lambda v: isinstance(v, (int, float)))
                ratio = numeric_mask.sum() / len(sample)
                if 0.05 < ratio < 0.95:
                    report["mixed_type_columns"].append(col)

        # Whitespace issues in string columns
        if pd.api.types.is_object_dtype(series):
            stripped = series.dropna().astype(str)
            ws_count = (stripped != stripped.str.strip()).sum()
            if ws_count > 0:
                report["whitespace_issues"][col] = int(ws_count)

    # ── Outliers (IQR for numeric cols) ──
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if col == target_col:
            continue
        clean_series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_series) < 10:
            continue
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_count = ((clean_series < lower) | (clean_series > upper)).sum()
        if outlier_count > 0:
            report["outliers"][col] = int(outlier_count)

    # ── Class distribution ──
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}

    return report


# ─────────────────────────────────────────────────────────────
# Universal Cleaning Pipeline
# ─────────────────────────────────────────────────────────────

def clean_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    sensitive_col: Optional[str] = None,
    remove_duplicates: bool = True,
    impute_missing: bool = True,
    handle_outliers: bool = True,
    scale_features: bool = False,
    balance_classes: bool = False,
    drop_id_columns: bool = True,
    drop_constant_columns: bool = True,
    fix_whitespace: bool = True,
    handle_infinite: bool = True,
    handle_datetime: bool = True,
    handle_mixed_types: bool = True,
    high_cardinality_method: str = "frequency",
    max_cardinality: int = 50,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Universal dataset cleaning pipeline.

    Handles any CSV dataset: mixed types, datetimes, ID columns, infinite
    values, constant columns, whitespace, high-cardinality categoricals,
    multiclass targets, and more.

    Parameters
    ----------
    df : pd.DataFrame
    target_col, sensitive_col : str, optional
        Protected from aggressive transforms (scaling, outlier capping, dropping).
    remove_duplicates : bool
    impute_missing : bool
    handle_outliers : bool
    scale_features : bool
    balance_classes : bool
    drop_id_columns : bool
        Remove auto-detected ID / index columns.
    drop_constant_columns : bool
        Remove columns with a single unique value.
    fix_whitespace : bool
        Strip leading/trailing whitespace in string columns.
    handle_infinite : bool
        Replace ±inf with NaN (then imputed if impute_missing=True).
    handle_datetime : bool
        Convert datetime columns to numeric features (year, month, day, etc.).
    handle_mixed_types : bool
        Coerce mixed-type columns to their dominant type.
    high_cardinality_method : str
        How to treat high-cardinality categoricals:
        ``"frequency"`` → replace with frequency counts,
        ``"top_n"`` → keep top *max_cardinality* values, rest → "Other",
        ``"drop"`` → remove the column.
    max_cardinality : int
        Threshold for "high cardinality" (default 50 unique values).

    Returns
    -------
    (cleaned_df, cleaning_report)
    """

    df_clean = df.copy()

    # Report accumulator
    report: Dict[str, Any] = {
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "duplicates_removed": 0,
        "missing_values_imputed": 0,
        "infinite_values_replaced": 0,
        "outliers_capped": 0,
        "features_scaled": 0,
        "smote_samples_added": 0,
        "columns_dropped": [],
        "datetime_features_created": [],
        "high_cardinality_encoded": [],
        "mixed_types_fixed": [],
        "whitespace_fixed": 0,
        "final_rows": 0,
        "final_columns": 0,
    }

    # Columns that must never be dropped or aggressively transformed
    _protected = set()
    if target_col and target_col in df_clean.columns:
        _protected.add(target_col)
    if sensitive_col and sensitive_col in df_clean.columns:
        _protected.add(sensitive_col)

    # ──────────────────────────────────────────────────────────
    # 0. Fix whitespace in string columns
    # ──────────────────────────────────────────────────────────
    if fix_whitespace:
        ws_fixed = 0
        for col in df_clean.select_dtypes(include=["object"]).columns:
            before = df_clean[col].copy()
            df_clean[col] = df_clean[col].astype(str).str.strip().replace("nan", np.nan)
            # Also collapse internal double-spaces
            df_clean[col] = df_clean[col].str.replace(r"\s+", " ", regex=True)
            ws_fixed += (before.fillna("") != df_clean[col].fillna("")).sum()
        report["whitespace_fixed"] = int(ws_fixed)

    # ──────────────────────────────────────────────────────────
    # 1. Replace infinite values → NaN
    # ──────────────────────────────────────────────────────────
    if handle_infinite:
        inf_count = 0
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            mask = np.isinf(df_clean[col])
            cnt = mask.sum()
            if cnt > 0:
                df_clean.loc[mask, col] = np.nan
                inf_count += int(cnt)
        report["infinite_values_replaced"] = inf_count

    # ──────────────────────────────────────────────────────────
    # 2. Drop ID / constant columns
    # ──────────────────────────────────────────────────────────
    cols_to_drop: List[str] = []

    if drop_id_columns:
        for col in df_clean.columns:
            if col in _protected:
                continue
            if _looks_like_id(df_clean[col], col):
                cols_to_drop.append(col)

    if drop_constant_columns:
        for col in df_clean.columns:
            if col in _protected or col in cols_to_drop:
                continue
            if _detect_constant_column(df_clean[col]):
                cols_to_drop.append(col)

    if cols_to_drop:
        df_clean.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        report["columns_dropped"] = cols_to_drop

    # ──────────────────────────────────────────────────────────
    # 3. Handle mixed-type columns
    # ──────────────────────────────────────────────────────────
    if handle_mixed_types:
        for col in list(df_clean.select_dtypes(include=["object"]).columns):
            if col in _protected:
                continue
            sample = df_clean[col].dropna().head(200)
            if len(sample) == 0:
                continue
            numeric_mask = pd.to_numeric(sample, errors="coerce").notna()
            ratio = numeric_mask.sum() / len(sample)
            if ratio > 0.5:
                # Dominant type is numeric – coerce whole column
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                report["mixed_types_fixed"].append(col)
            # If ratio ≤ 0.5 we keep as categorical (non-numeric values → as-is)

    # ──────────────────────────────────────────────────────────
    # 4. Handle datetime columns (convert to numeric features)
    # ──────────────────────────────────────────────────────────
    if handle_datetime:
        for col in list(df_clean.columns):
            if col in _protected:
                continue
            series = df_clean[col]
            if _is_datetime_dtype(series) or (
                pd.api.types.is_object_dtype(series) and _looks_like_datetime(series)
            ):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        dt = pd.to_datetime(df_clean[col], errors="coerce")
                    if dt.notna().sum() > 0:
                        df_clean[f"{col}_year"] = dt.dt.year
                        df_clean[f"{col}_month"] = dt.dt.month
                        df_clean[f"{col}_day"] = dt.dt.day
                        df_clean[f"{col}_dayofweek"] = dt.dt.dayofweek
                        report["datetime_features_created"].append(col)
                        df_clean.drop(columns=[col], inplace=True)
                except Exception:
                    pass  # Leave column as-is if conversion fails

    # ──────────────────────────────────────────────────────────
    # 5. Remove duplicate rows
    # ──────────────────────────────────────────────────────────
    if remove_duplicates:
        before = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        report["duplicates_removed"] = before - len(df_clean)

    # ──────────────────────────────────────────────────────────
    # 6. Impute missing values (type-aware)
    # ──────────────────────────────────────────────────────────
    if impute_missing:
        missing_count = 0

        for col in df_clean.columns:
            n_missing = df_clean[col].isnull().sum()
            if n_missing == 0:
                continue

            # If >95 % missing and not protected → drop column
            if (n_missing / len(df_clean)) > 0.95 and col not in _protected:
                df_clean.drop(columns=[col], inplace=True)
                report["columns_dropped"].append(col)
                continue

            missing_count += int(n_missing)

            if _is_numeric_dtype(df_clean[col]):
                # Numeric → median (robust to outliers)
                fill = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(fill)
            else:
                # Categorical → mode; fall back to "Unknown"
                mode_vals = df_clean[col].mode()
                fill = mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown"
                df_clean[col] = df_clean[col].fillna(fill)

        report["missing_values_imputed"] = missing_count

    # ──────────────────────────────────────────────────────────
    # 7. Handle high-cardinality categorical columns
    # ──────────────────────────────────────────────────────────
    for col in list(df_clean.select_dtypes(include=["object", "category"]).columns):
        if col in _protected:
            continue
        n_unique = df_clean[col].nunique()
        if n_unique <= max_cardinality:
            continue

        report["high_cardinality_encoded"].append(col)

        if high_cardinality_method == "frequency":
            freq_map = df_clean[col].value_counts(normalize=True).to_dict()
            df_clean[col] = df_clean[col].map(freq_map).astype(float)
        elif high_cardinality_method == "top_n":
            top_values = df_clean[col].value_counts().head(max_cardinality).index.tolist()
            df_clean[col] = df_clean[col].where(df_clean[col].isin(top_values), other="Other")
        elif high_cardinality_method == "drop":
            df_clean.drop(columns=[col], inplace=True)
            report["columns_dropped"].append(col)
        # else: leave as-is

    # ──────────────────────────────────────────────────────────
    # 8. Handle outliers (IQR capping on numeric columns)
    # ──────────────────────────────────────────────────────────
    if handle_outliers:
        outlier_count = 0
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            if col in _protected:
                continue
            clean_series = df_clean[col].dropna()
            if len(clean_series) < 10:
                continue
            Q1 = clean_series.quantile(0.25)
            Q3 = clean_series.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            mask = (df_clean[col] < lower) | (df_clean[col] > upper)
            outlier_count += int(mask.sum())
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
        report["outliers_capped"] = outlier_count

    # ──────────────────────────────────────────────────────────
    # 9. Scale numeric features
    # ──────────────────────────────────────────────────────────
    if scale_features:
        num_cols = [
            c for c in df_clean.select_dtypes(include=[np.number]).columns
            if c not in _protected
        ]
        if num_cols:
            scaler = StandardScaler()
            df_clean[num_cols] = scaler.fit_transform(df_clean[num_cols])
            report["features_scaled"] = len(num_cols)

    # ──────────────────────────────────────────────────────────
    # 10. Balance classes (SMOTE – binary & multiclass)
    # ──────────────────────────────────────────────────────────
    if balance_classes and target_col and target_col in df_clean.columns and _SMOTE_AVAILABLE:
        y = df_clean[target_col].copy()
        X = df_clean.drop(columns=[target_col])

        class_counts = y.value_counts()
        is_imbalanced = class_counts.min() < class_counts.max() * 0.8

        if is_imbalanced and len(class_counts) >= 2:
            # Encode target if non-numeric
            target_le = None
            if not _is_numeric_dtype(y):
                target_le = LabelEncoder()
                y = pd.Series(target_le.fit_transform(y), name=target_col)

            # Identify which columns are still categorical
            cat_indices = [
                i for i, c in enumerate(X.columns)
                if _is_categorical_dtype(X[c])
            ]

            # One-hot encode remaining categoricals for SMOTE
            X_encoded = pd.get_dummies(X, drop_first=True)

            try:
                min_samples = class_counts.min()
                k = min(5, max(1, min_samples - 1))
                if k < 1:
                    raise ValueError("Too few samples in minority class for SMOTE")

                smote = SMOTE(random_state=42, k_neighbors=k)
                X_res, y_res = smote.fit_resample(X_encoded, y)

                df_clean = pd.DataFrame(X_res, columns=X_encoded.columns)
                if target_le is not None:
                    df_clean[target_col] = target_le.inverse_transform(y_res.astype(int))
                else:
                    df_clean[target_col] = y_res
                report["smote_samples_added"] = len(y_res) - len(y)
            except Exception as e:
                warnings.warn(f"SMOTE balancing skipped: {e}")

    # ──────────────────────────────────────────────────────────
    # Finalize
    # ──────────────────────────────────────────────────────────
    report["final_rows"] = len(df_clean)
    report["final_columns"] = len(df_clean.columns)

    return df_clean, report


# ─────────────────────────────────────────────────────────────
# Summary / helpers
# ─────────────────────────────────────────────────────────────

def get_cleaning_summary(report: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary from a cleaning report.

    Parameters
    ----------
    report : dict
        Cleaning report from ``clean_dataset()``.

    Returns
    -------
    str
    """
    lines = [
        "╔══════════════════════════════════════╗",
        "║        Cleaning Summary              ║",
        "╚══════════════════════════════════════╝",
        f"  Original size   : {report['original_rows']:,} rows × {report.get('original_columns', '?')} cols",
        f"  Final size      : {report['final_rows']:,} rows × {report.get('final_columns', '?')} cols",
        f"  Duplicates removed        : {report['duplicates_removed']:,}",
        f"  Missing values imputed    : {report['missing_values_imputed']:,}",
        f"  Infinite values replaced  : {report.get('infinite_values_replaced', 0):,}",
        f"  Outliers capped           : {report['outliers_capped']:,}",
        f"  Features scaled           : {report['features_scaled']:,}",
        f"  SMOTE samples added       : {report['smote_samples_added']:,}",
        f"  Whitespace fixes          : {report.get('whitespace_fixed', 0):,}",
    ]
    if report.get("columns_dropped"):
        lines.append(f"  Columns dropped           : {', '.join(report['columns_dropped'])}")
    if report.get("datetime_features_created"):
        lines.append(f"  Datetime features created : {', '.join(report['datetime_features_created'])}")
    if report.get("high_cardinality_encoded"):
        lines.append(f"  High-card cols encoded    : {', '.join(report['high_cardinality_encoded'])}")
    if report.get("mixed_types_fixed"):
        lines.append(f"  Mixed-type cols fixed     : {', '.join(report['mixed_types_fixed'])}")
    return "\n".join(lines)


def has_quality_issues(quality_report: Dict[str, Any]) -> bool:
    """
    Return True if the quality report contains any actionable issues.

    Parameters
    ----------
    quality_report : dict
        Report from ``analyze_data_quality()``.
    """
    checks = [
        len(quality_report.get("missing_values", {})) > 0,
        quality_report.get("duplicates", 0) > 0,
        len(quality_report.get("outliers", {})) > 0,
        len(quality_report.get("infinite_values", {})) > 0,
        len(quality_report.get("constant_columns", [])) > 0,
        len(quality_report.get("id_columns", [])) > 0,
        len(quality_report.get("high_cardinality", [])) > 0,
        len(quality_report.get("mixed_type_columns", [])) > 0,
        len(quality_report.get("whitespace_issues", {})) > 0,
    ]

    # Class imbalance (works for binary AND multiclass)
    dist = quality_report.get("class_distribution")
    if dist:
        counts = list(dist.values())
        if len(counts) >= 2:
            ratio = min(counts) / max(counts) if max(counts) > 0 else 1
            checks.append(ratio < 0.5)

    return any(checks)
