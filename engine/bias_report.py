def generate_bias_flag(fairness_scores, dp_threshold=0.1, di_threshold=0.8):
    dp_flag = fairness_scores["demographic_parity_difference"] > dp_threshold
    di_flag = fairness_scores["disparate_impact_ratio"] < di_threshold

    return dp_flag or di_flag
