from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_baseline(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, y_pred, accuracy
