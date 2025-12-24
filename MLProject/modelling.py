import os
import shutil
import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.models.signature import infer_signature

DATA_DIR = "Telco-Customer-Churn_clean"
MODEL_DIR = "logreg_model"

X_train = joblib.load(os.path.join(DATA_DIR, "X_train.pkl"))
X_test  = joblib.load(os.path.join(DATA_DIR, "X_test.pkl"))
y_train = joblib.load(os.path.join(DATA_DIR, "y_train.pkl"))
y_test  = joblib.load(os.path.join(DATA_DIR, "y_test.pkl"))

with mlflow.start_run(run_name="LogReg_CI_Training"):


    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="Yes")
    rec  = recall_score(y_test, y_pred, pos_label="Yes")
    f1   = f1_score(y_test, y_pred, pos_label="Yes")

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.save_model(
        sk_model=model,
        path=MODEL_DIR,
        signature=signature,
        input_example=X_train[:5]
    )

    mlflow.log_artifacts(MODEL_DIR, artifact_path=MODEL_DIR)

    print("✅ CI training & MLflow artifact logging selesai.")