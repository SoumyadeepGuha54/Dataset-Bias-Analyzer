"""
Enhanced bias analysis pipeline with multi-model support and data cleaning.
"""

import pandas as pd
import numpy as np
from engine.data_processor import prepare_data
from engine.model_trainer import train_baseline, train_multiple_models, get_feature_importance
from engine.fairness_metrics import compute_group_metrics, compute_fairness_scores
from engine.bias_report import generate_bias_flag
from engine.proxy_detector import detect_proxies
from engine.data_cleaner import analyze_data_quality, clean_dataset


def run_bias_engine(df, target_col, sensitive_col, models=None, cleaning_config=None):
    """
    Enhanced bias analysis pipeline with multi-model support and optional data cleaning.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset
    target_col : str
        Name of the label column (binary 0/1)
    sensitive_col : str
        Name of the sensitive attribute column
    models : list of str, optional
        List of models to train. Options: 'logistic_regression', 'random_forest', 
        'svm', 'xgboost'. If None, trains only logistic regression (backward compatible).
    cleaning_config : dict, optional
        Data cleaning configuration with keys:
        - remove_duplicates: bool
        - impute_missing: bool
        - handle_outliers: bool
        - scale_features: bool
        - balance_classes: bool
        If None, no cleaning is performed.

    Returns
    -------
    dict
        Enhanced results containing:
        - models: dict with results for each model (predictions, metrics, fairness_scores, bias_detected)
        - proxy_variables: dict of detected proxy variables
        - cleaning_report: dict with cleaning statistics (if cleaning was performed)
        - data_quality: dict with quality analysis
        - feature_names: list of feature names after encoding
    """
    # Initialize results
    results = {
        'models': {},
        'proxy_variables': {},
        'cleaning_report': None,
        'data_quality': None,
    }
    
    # --- Step 1: Data Quality Analysis ---
    results['data_quality'] = analyze_data_quality(df, target_col)
    
    # --- Step 2: Data Cleaning (if requested) ---
    df_processed = df.copy()
    if cleaning_config:
        df_processed, cleaning_report = clean_dataset(
            df_processed,
            target_col=target_col,
            sensitive_col=sensitive_col,
            **cleaning_config
        )
        results['cleaning_report'] = cleaning_report
    
    # --- Step 3: Proxy Detection (runs on data before encoding) ---
    results['proxy_variables'] = detect_proxies(
        df_processed.drop(columns=[target_col], errors="ignore"),
        sensitive_col,
    )

    # --- Step 4: Data Preparation ---
    X_train, X_test, y_train, y_test, sensitive_test = prepare_data(
        df_processed, target_col, sensitive_col
    )
    
    # Store feature names (after encoding)
    if hasattr(X_train, 'columns'):
        results['feature_names'] = X_train.columns.tolist()
    else:
        results['feature_names'] = [f'feature_{i}' for i in range(X_train.shape[1])]

    # --- Step 5: Model Training ---
    if models is None or len(models) == 0:
        # Backward compatibility: train only baseline logistic regression
        model, y_pred, accuracy = train_baseline(X_train, y_train, X_test, y_test)
        
        # Compute fairness metrics
        group_metrics = compute_group_metrics(y_test, y_pred, sensitive_test)
        fairness_scores = compute_fairness_scores(group_metrics)
        bias_flag = generate_bias_flag(fairness_scores)
        
        # Return old format for backward compatibility
        return {
            "accuracy": accuracy,
            "group_metrics": group_metrics,
            "fairness_scores": fairness_scores,
            "bias_detected": bias_flag,
            "proxy_variables": results['proxy_variables'],
        }
    
    # Multi-model training
    model_results = train_multiple_models(X_train, y_train, X_test, y_test, models)
    
    # --- Step 6: Fairness Evaluation for Each Model ---
    for model_name, model_data in model_results.items():
        y_pred = model_data['predictions']
        
        # Compute fairness metrics
        group_metrics = compute_group_metrics(y_test, y_pred, sensitive_test)
        fairness_scores = compute_fairness_scores(group_metrics)
        bias_flag = generate_bias_flag(fairness_scores)
        
        # Get feature importance
        feature_importance = get_feature_importance(
            model_data['model'],
            results['feature_names'],
            model_name
        )
        
        # Store results for this model
        results['models'][model_name] = {
            'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
            'probabilities': model_data['probabilities'].tolist() if model_data['probabilities'] is not None else None,
            'metrics': model_data['metrics'],
            'group_metrics': group_metrics,
            'fairness_scores': fairness_scores,
            'bias_detected': bias_flag,
            'feature_importance': feature_importance
        }
    
    # Store test data for visualizations
    results['test_data'] = {
        'y_test': y_test.tolist() if hasattr(y_test, 'tolist') else y_test,
        'sensitive_test': sensitive_test.tolist() if hasattr(sensitive_test, 'tolist') else sensitive_test
    }
    
    return results


def run_bias_engine_legacy(df, target_col, sensitive_col):
    """
    Legacy bias analysis pipeline (backward compatibility).
    
    This is the original function signature maintained for compatibility.
    Use run_bias_engine() with models parameter for enhanced functionality.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str – name of the label column (binary 0/1)
    sensitive_col : str – name of the sensitive attribute column

    Returns
    -------
    dict with keys: accuracy, group_metrics, fairness_scores, bias_detected, proxy_variables
    """
    return run_bias_engine(df, target_col, sensitive_col, models=None, cleaning_config=None)


if __name__ == "__main__":
    # Example local run
    # df = pd.read_csv("data.csv")
    # result = run_bias_engine(df, "target", "gender")
    # print(result)
    pass
