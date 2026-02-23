import pandas as pd
from engine.data_processor import prepare_data
from engine.model_trainer import train_baseline
from engine.fairness_metrics import compute_group_metrics, compute_fairness_scores
from engine.bias_report import generate_bias_flag
from engine.proxy_detector import detect_proxies


def run_bias_engine(df, target_col, sensitive_col):
    """
    Full bias analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str – name of the label column (binary 0/1)
    sensitive_col : str – name of the sensitive attribute column

    Returns
    -------
    dict with keys: accuracy, group_metrics, fairness_scores, bias_detected, proxy_variables
    """
    # --- Proxy detection (runs on raw data before encoding) ---
    proxy_variables = detect_proxies(
        df.drop(columns=[target_col], errors="ignore"),
        sensitive_col,
    )

    # --- Data preparation ---
    X_train, X_test, y_train, y_test, sensitive_test = prepare_data(
        df, target_col, sensitive_col
    )

    # --- Model training ---
    model, y_pred, accuracy = train_baseline(
        X_train, y_train, X_test, y_test
    )

    # --- Fairness evaluation ---
    group_metrics = compute_group_metrics(
        y_test, y_pred, sensitive_test
    )

    fairness_scores = compute_fairness_scores(group_metrics)
    bias_flag = generate_bias_flag(fairness_scores)

    return {
        "accuracy": accuracy,
        "group_metrics": group_metrics,
        "fairness_scores": fairness_scores,
        "bias_detected": bias_flag,
        "proxy_variables": proxy_variables,
    }


if __name__ == "__main__":
    # Example local run
    # df = pd.read_csv("data.csv")
    # result = run_bias_engine(df, "target", "gender")
    # print(result)
    pass
