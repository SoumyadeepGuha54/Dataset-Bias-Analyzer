import pandas as pd
import numpy as np

def compute_group_metrics(y_true, y_pred, sensitive):
    """
    Compute group-level fairness metrics.
    
    Note: This function is designed for binary classification.
    For multiclass problems, it uses one-vs-rest approach (class 1 vs others).
    """
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "sensitive": sensitive
    })

    groups = df["sensitive"].unique()
    results = {}
    
    # Detect if binary or multiclass
    unique_true = df["y_true"].unique()
    unique_pred = df["y_pred"].unique()
    
    # For multiclass, convert to binary (positive class vs rest)
    # Assume the highest class value is the "positive" class
    if len(unique_true) > 2 or len(unique_pred) > 2:
        positive_class = max(df["y_true"].unique())
        df["y_true_binary"] = (df["y_true"] == positive_class).astype(int)
        df["y_pred_binary"] = (df["y_pred"] == positive_class).astype(int)
    else:
        df["y_true_binary"] = df["y_true"]
        df["y_pred_binary"] = df["y_pred"]

    for g in groups:
        group_df = df[df["sensitive"] == g]
        approval_rate = group_df["y_pred_binary"].mean()

        positives = group_df[group_df["y_true_binary"] == 1]
        tpr = positives["y_pred_binary"].mean() if len(positives) > 0 else 0

        negatives = group_df[group_df["y_true_binary"] == 0]
        fpr = negatives["y_pred_binary"].mean() if len(negatives) > 0 else 0

        results[g] = {
            "approval_rate": approval_rate,
            "tpr": tpr,
            "fpr": fpr
        }

    return results

def compute_fairness_scores(group_metrics):
    approval_rates = [v["approval_rate"] for v in group_metrics.values()]
    tprs = [v["tpr"] for v in group_metrics.values()]
    fprs = [v["fpr"] for v in group_metrics.values()]

    dp_diff = max(approval_rates) - min(approval_rates)
    eo_diff = max(tprs) - min(tprs)
    eod_diff = max(fprs) - min(fprs)

    di_ratio = 0 if min(approval_rates) == 0 else min(approval_rates) / max(approval_rates)

    return {
        "demographic_parity_difference": dp_diff,
        "equal_opportunity_difference": eo_diff,
        "equalized_odds_difference": eod_diff,
        "disparate_impact_ratio": di_ratio
    }
