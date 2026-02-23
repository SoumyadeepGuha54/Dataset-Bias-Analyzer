import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df, target_col, sensitive_col):
    if target_col not in df.columns:
        raise ValueError("Target column not found.")
    if sensitive_col not in df.columns:
        raise ValueError("Sensitive column not found.")

    df = df.dropna(subset=[target_col, sensitive_col])

    y = df[target_col]
    sensitive = df[sensitive_col]
    X = df.drop(columns=[target_col])

    # Auto-encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sensitive_test = sensitive.loc[X_test.index]

    return X_train, X_test, y_train, y_test, sensitive_test
