"""
Data quality analysis and cleaning module for Dataset Bias Analyzer.

Provides functions to:
- Detect data quality issues (missing values, duplicates, outliers, class imbalance)
- Clean datasets with configurable options
- Generate cleaning reports with before/after statistics
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def analyze_data_quality(df, target_col=None):
    """
    Analyze dataset for quality issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target_col : str, optional
        Name of target column for class imbalance analysis
    
    Returns
    -------
    dict
        Report containing:
        - missing_values: dict of {column: count} for columns with missing data
        - duplicates: number of duplicate rows
        - outliers: dict of {column: count} for numerical columns with outliers
        - dtypes: dict of column data types
        - class_distribution: dict of class counts if target_col provided
        - total_rows: int
        - total_columns: int
    """
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicates": 0,
        "outliers": {},
        "dtypes": {},
        "class_distribution": None
    }
    
    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = {col: int(count) for col, count in missing.items() if count > 0}
    
    # Duplicates
    report["duplicates"] = int(df.duplicated().sum())
    
    # Data types
    report["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}
    
    # Outliers (IQR method for numerical columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if col == target_col:
            continue
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        if outlier_count > 0:
            report["outliers"][col] = int(outlier_count)
    
    # Class distribution
    if target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in class_counts.items()}
    
    return report


def clean_dataset(df, target_col=None, sensitive_col=None,
                  remove_duplicates=True,
                  impute_missing=True,
                  handle_outliers=True,
                  scale_features=False,
                  balance_classes=False):
    """
    Clean dataset based on specified options.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target_col : str, optional
        Name of target column (excluded from scaling/outlier handling)
    sensitive_col : str, optional
        Name of sensitive attribute column (excluded from scaling/outlier handling)
    remove_duplicates : bool, default=True
        Remove exact duplicate rows
    impute_missing : bool, default=True
        Impute missing values (mean for numerical, mode for categorical)
    handle_outliers : bool, default=True
        Cap outliers using IQR method
    scale_features : bool, default=False
        Standardize numerical features (mean=0, std=1)
    balance_classes : bool, default=False
        Apply SMOTE to balance minority class (requires target_col)
    
    Returns
    -------
    tuple
        (cleaned_df, cleaning_report)
        - cleaned_df: pd.DataFrame with cleaning applied
        - cleaning_report: dict with statistics about cleaning operations
    """
    df_clean = df.copy()
    report = {
        "duplicates_removed": 0,
        "missing_values_imputed": 0,
        "outliers_capped": 0,
        "features_scaled": 0,
        "smote_samples_added": 0,
        "original_rows": len(df),
        "final_rows": 0
    }
    
    # Identify columns to exclude from certain operations
    exclude_cols = []
    if target_col:
        exclude_cols.append(target_col)
    if sensitive_col:
        exclude_cols.append(sensitive_col)
    
    # Step 1: Remove duplicates
    if remove_duplicates:
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        report["duplicates_removed"] = initial_rows - len(df_clean)
    
    # Step 2: Impute missing values
    if impute_missing:
        missing_count = 0
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in [np.float64, np.int64]:
                    # Numerical: use median
                    fill_value = df_clean[col].median()
                    missing_count += df_clean[col].isnull().sum()
                    df_clean[col].fillna(fill_value, inplace=True)
                else:
                    # Categorical: use mode
                    fill_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                    missing_count += df_clean[col].isnull().sum()
                    df_clean[col].fillna(fill_value, inplace=True)
        report["missing_values_imputed"] = int(missing_count)
    
    # Step 3: Handle outliers (IQR method)
    if handle_outliers:
        outlier_count = 0
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if col in exclude_cols:
                continue
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            outlier_count += outliers
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        report["outliers_capped"] = int(outlier_count)
    
    # Step 4: Scale features
    if scale_features:
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_scale = [col for col in numerical_cols if col not in exclude_cols]
        
        if cols_to_scale:
            scaler = StandardScaler()
            df_clean[cols_to_scale] = scaler.fit_transform(df_clean[cols_to_scale])
            report["features_scaled"] = len(cols_to_scale)
    
    # Step 5: Balance classes using SMOTE
    if balance_classes and target_col and target_col in df_clean.columns:
        # Separate features and target
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Check if classes are imbalanced
        class_counts = y.value_counts()
        if len(class_counts) == 2 and class_counts.min() < class_counts.max():
            # Need to encode categorical features for SMOTE
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            try:
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X_encoded, y)
                
                # Reconstruct dataframe
                df_clean = pd.DataFrame(X_resampled, columns=X_encoded.columns)
                df_clean[target_col] = y_resampled
                
                report["smote_samples_added"] = len(y_resampled) - len(y)
            except Exception as e:
                # SMOTE may fail with certain data conditions
                print(f"SMOTE failed: {e}. Skipping class balancing.")
    
    report["final_rows"] = len(df_clean)
    
    return df_clean, report


def get_cleaning_summary(report):
    """
    Generate human-readable summary from cleaning report.
    
    Parameters
    ----------
    report : dict
        Cleaning report from clean_dataset()
    
    Returns
    -------
    str
        Formatted summary text
    """
    summary_lines = [
        f"Cleaning Summary:",
        f"  Original rows: {report['original_rows']}",
        f"  Final rows: {report['final_rows']}",
        f"  Duplicates removed: {report['duplicates_removed']}",
        f"  Missing values imputed: {report['missing_values_imputed']}",
        f"  Outliers capped: {report['outliers_capped']}",
        f"  Features scaled: {report['features_scaled']}",
        f"  SMOTE samples added: {report['smote_samples_added']}"
    ]
    return "\n".join(summary_lines)


def has_quality_issues(quality_report):
    """
    Check if dataset has any quality issues.
    
    Parameters
    ----------
    quality_report : dict
        Report from analyze_data_quality()
    
    Returns
    -------
    bool
        True if any quality issues detected
    """
    has_missing = len(quality_report.get("missing_values", {})) > 0
    has_duplicates = quality_report.get("duplicates", 0) > 0
    has_outliers = len(quality_report.get("outliers", {})) > 0
    
    # Check class imbalance (if distribution exists)
    has_imbalance = False
    if quality_report.get("class_distribution"):
        counts = list(quality_report["class_distribution"].values())
        if len(counts) == 2:
            ratio = min(counts) / max(counts) if max(counts) > 0 else 1
            has_imbalance = ratio < 0.5  # Flag if minority class is less than 50% of majority
    
    return has_missing or has_duplicates or has_outliers or has_imbalance
