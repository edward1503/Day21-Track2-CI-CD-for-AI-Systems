import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    # TODO 1: Đọc dữ liệu
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 2: Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    from pathlib import Path
    import tempfile
    
    if os.getenv("CI"):
        mlflow_dir = Path(tempfile.mkdtemp(prefix="mlruns_"))
    else:
        mlflow_dir = Path("mlruns").resolve()

    mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("day21")

    with mlflow.start_run():
        # TODO 3: Log params
        mlflow.log_params(params)

        # TODO 4: Khởi tạo và train
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Đánh giá
        preds = model.predict(X_eval)
        acc = accuracy_score(y_eval, preds)
        f1 = f1_score(y_eval, preds, average="weighted")

        # TODO 6: Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In kết quả
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

        # TODO 8: Lưu metrics.json
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1}, f)

        # TODO 9: Lưu model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Trả về accuracy
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)