import os
import joblib
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "Telco-Customer-Churn_clean"

X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test  = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test  = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

with mlflow.start_run():

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, pos_label="Yes"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, pos_label="Yes"))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, pos_label="Yes"))

    mlflow.sklearn.log_model(model, "model")

    print("CI training selesai.")
