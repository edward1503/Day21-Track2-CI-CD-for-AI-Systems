import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

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
    
    if os.getenv("MLFLOW_TRACKING_URI"):
        # Bonus 1: Use DagsHub tracking URI
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    else:
        if os.getenv("CI"):
            mlflow_dir = Path(tempfile.mkdtemp(prefix="mlruns_"))
        else:
            mlflow_dir = Path("mlruns").resolve()
        mlflow.set_tracking_uri(mlflow_dir.as_uri())
    mlflow.set_experiment("day21")

    with mlflow.start_run():
        # TODO 3: Log params
        mlflow.log_params(params)

        # Bonus 5: Cảnh báo lệch lạc dữ liệu
        class_dist = y_train.value_counts(normalize=True).to_dict()
        for cls, pct in class_dist.items():
            if pct < 0.1:
                print(f"WARNING: Lớp {cls} chỉ chiếm {pct:.2%} (< 10%) tổng số mẫu. Dữ liệu bị lệch!")

        # Bonus 2: Lựa chọn thuật toán
        model_type = params.get("model_type", "random_forest")
        model_params = params.get(model_type, params) # fallback for old params format

        if model_type == "random_forest":
            model = RandomForestClassifier(**model_params, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**model_params, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**model_params, random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Không hỗ trợ thuật toán: {model_type}")

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

        # Bonus 3: Báo cáo hiệu suất tự động
        cm = confusion_matrix(y_eval, preds)
        cr = classification_report(y_eval, preds, zero_division=0)
        
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report.txt", "w", encoding="utf-8") as f:
            f.write(f"Model Type: {model_type}\n\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm))
            f.write("\n\nClassification Report:\n")
            f.write(cr)

        # TODO 8: Lưu metrics.json (Kèm Bonus 5)
        with open("outputs/metrics.json", "w") as f:
            json.dump({"accuracy": acc, "f1_score": f1, "train_distribution": class_dist}, f)

        # TODO 9: Lưu model.pkl
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

    # TODO 10: Trả về accuracy
    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)