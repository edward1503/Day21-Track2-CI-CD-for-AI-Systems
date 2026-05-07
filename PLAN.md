# PLAN - Day 21: CI/CD for AI Systems

## Tổng quan

Lab gồm 3 bước:
- **Bước 1**: Thực nghiệm cục bộ + MLflow tracking
- **Bước 2**: Pipeline CI/CD với GitHub Actions + DVC + Cloud VM
- **Bước 3**: Continuous training khi có dữ liệu mới

Cloud provider khuyến nghị: **GCP** (hướng dẫn mặc định trong lab). Có thể dùng AWS hoặc Azure.

> **LƯU Ý QUAN TRỌNG**: Workflow hiện tại trigger trên branch `main`, nhưng repo đang dùng branch `master`. Cần đổi branch trigger trong `mlops.yml` thành `master` hoặc đổi tên branch Git sang `main`.

---

## Chuẩn bị môi trường (một lần duy nhất)

```powershell
# 1. Tạo và kích hoạt venv
python -m venv .venv
.venv\Scripts\activate   # Windows

# 2. Cài thư viện
pip install -r requirements.txt

# 3. Tạo dữ liệu
python generate_data.py
```

Kết quả mong đợi:
```
train_phase1.csv : 2998 mẫu
eval.csv         :  500 mẫu
train_phase2.csv : 2998 mẫu
```

Kiểm tra:
```powershell
ls data/
```

---

## BƯỚC 1 - Thực nghiệm cục bộ với MLflow

### 1.1 Set biến môi trường MLflow

**Windows PowerShell:**
```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
$env:MLFLOW_ARTIFACT_ROOT = "./mlartifacts"
```

**Lưu ý**: Biến này mất khi đóng terminal. Để tiện hơn, tạo file `.env` và load mỗi lần (hoặc thêm vào PowerShell profile).

### 1.2 Hoàn thiện `src/train.py`

Thay toàn bộ phần TODO trong `src/train.py` bằng code hoàn chỉnh:

```python
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
```

### 1.3 Chạy ít nhất 3 thí nghiệm

Lần 1 (mặc định: n_estimators=100, max_depth=5, min_samples_split=2):
```powershell
python src/train.py
```

Sửa `params.yaml` → Lần 2 (n_estimators=50, max_depth=3):
```yaml
n_estimators: 50
max_depth: 3
min_samples_split: 2
```
```powershell
python src/train.py
```

Sửa `params.yaml` → Lần 3 (n_estimators=200, max_depth=10):
```yaml
n_estimators: 200
max_depth: 10
min_samples_split: 5
```
```powershell
python src/train.py
```

Gợi ý: chạy thêm 1-2 lần nữa để có nhiều dữ liệu so sánh hơn.

### 1.4 Xem MLflow UI

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Mở trình duyệt → http://localhost:5000

- Sắp xếp theo `accuracy` giảm dần
- So sánh nhiều lần chạy (chọn nhiều run → "Compare")
- Ghi lại bộ siêu tham số tốt nhất

### 1.5 Cập nhật params.yaml với bộ tốt nhất

Trước khi sang Bước 2, đặt bộ siêu tham số tốt nhất vào `params.yaml`.

### Checklist Bước 1

- [ ] `src/train.py` chạy thành công không có lỗi
- [ ] `outputs/metrics.json` tồn tại và có `accuracy` + `f1_score`
- [ ] `models/model.pkl` tồn tại
- [ ] MLflow UI hiển thị ≥ 3 lần chạy với siêu tham số khác nhau
- [ ] `params.yaml` đã cập nhật với bộ tốt nhất
- [ ] **Chụp màn hình MLflow UI** (bắt buộc nộp bài)

---

## BƯỚC 2 - Pipeline CI/CD tự động

### 2.1 Tạo Cloud Storage Bucket (GCP)

```bash
export PROJECT=<YOUR_GCP_PROJECT_ID>
export BUCKET=<YOUR_UNIQUE_BUCKET_NAME>   # tên toàn cầu duy nhất

# Kích hoạt API (nếu chưa có)
gcloud services enable storage.googleapis.com --project $PROJECT

# Tạo bucket
gsutil mb -p $PROJECT -l us-central1 gs://$BUCKET
```

**AWS thay thế:**
```bash
aws s3 mb s3://$BUCKET --region us-east-1
```

**Azure thay thế:**
```bash
az storage container create --name $CONTAINER --account-name <STORAGE_ACCOUNT>
```

### 2.2 Tạo Service Account và lấy credentials (GCP)

```bash
# Tạo service account
gcloud iam service-accounts create mlops-lab-sa \
  --display-name "MLOps Lab SA" \
  --project $PROJECT

# Cấp quyền objectAdmin CHỈ trên bucket (không phải toàn project)
gsutil iam ch \
  serviceAccount:mlops-lab-sa@$PROJECT.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://$BUCKET

# Xuất file key JSON
gcloud iam service-accounts keys create sa-key.json \
  --iam-account mlops-lab-sa@$PROJECT.iam.gserviceaccount.com
```

> **QUAN TRỌNG**: `sa-key.json` **KHÔNG** được commit lên git. File này đã có trong `.gitignore`.

### 2.3 Cài đặt và cấu hình DVC

```bash
dvc init

# GCP:
dvc remote add -d myremote gs://$BUCKET/dvc
dvc remote modify myremote credentialpath sa-key.json

# AWS: dvc remote add -d myremote s3://$BUCKET/dvc
# Azure: dvc remote add -d myremote azure://mycontainer/dvc
#        dvc remote modify myremote connection_string "<CONNECTION_STRING>"

# Track các file dữ liệu
dvc add data/train_phase1.csv
dvc add data/eval.csv
dvc add data/train_phase2.csv

# Commit file con trỏ DVC (KHÔNG phải file CSV)
git add data/train_phase1.csv.dvc data/eval.csv.dvc data/train_phase2.csv.dvc \
        .gitignore .dvc/config
git commit -m "feat: track datasets with DVC"

# Push dữ liệu lên cloud storage
dvc push
```

Xác nhận: vào Cloud Storage Console → thấy file dưới prefix `dvc/`.

### 2.4 Tạo VM trên Cloud (GCP)

```bash
gcloud compute instances create mlops-serve \
  --zone=us-central1-a \
  --machine-type=e2-small \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --tags=mlops-serve \
  --project $PROJECT

# Mở cổng 8000
gcloud compute firewall-rules create allow-mlops-serve \
  --allow=tcp:8000 \
  --target-tags=mlops-serve \
  --project $PROJECT

# Lấy IP công khai (lưu lại)
gcloud compute instances describe mlops-serve \
  --zone=us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

### 2.5 Cấu hình VM (một lần, thủ công)

SSH vào VM:
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a
```

Bên trong VM:
```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install fastapi uvicorn scikit-learn joblib google-cloud-storage
mkdir -p ~/models ~/src
exit
```

Copy file key lên VM (chạy trên máy local):
```bash
gcloud compute scp sa-key.json mlops-serve:~/sa-key.json --zone=us-central1-a
```

### 2.6 Hoàn thiện `src/serve.py`

```python
from fastapi import FastAPI, HTTPException
from google.cloud import storage
import joblib
import os

app = FastAPI()

GCS_BUCKET = os.environ["GCS_BUCKET"]
GCS_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(GCS_MODEL_KEY)
    blob.download_to_filename(MODEL_PATH)
    print(f"Model downloaded to {MODEL_PATH}")


download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest:
    features: list[float]

from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = model.predict([req.features])[0]
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

Copy serve.py lên VM:
```bash
gcloud compute scp src/serve.py mlops-serve:~/src/serve.py --zone=us-central1-a
```

### 2.7 Cấu hình systemd service trên VM

SSH vào VM:
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a
```

Tạo service file (thay `<YOUR_BUCKET_NAME>` và `<YOUR_USER>`):
```bash
sudo tee /etc/systemd/system/mlops-serve.service > /dev/null <<EOF
[Unit]
Description=MLOps Model Inference Server
After=network.target

[Service]
User=$USER
WorkingDirectory=/home/$USER
Environment="GCS_BUCKET=<YOUR_BUCKET_NAME>"
Environment="GOOGLE_APPLICATION_CREDENTIALS=/home/$USER/sa-key.json"
ExecStart=/usr/bin/python3 /home/$USER/src/serve.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable mlops-serve
exit
```

> **Lưu ý**: Chưa cần `systemctl start` lúc này. Service chỉ khởi động được sau khi pipeline chạy lần đầu và model.pkl đã có trên GCS.

### 2.8 Tạo SSH key để GitHub Actions deploy

Chạy trên máy local:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/mlops_deploy -N "" -C "github-actions-deploy"
```

Thêm public key vào VM:
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a \
  --command "echo '$(cat ~/.ssh/mlops_deploy.pub)' >> ~/.ssh/authorized_keys"
```

### 2.9 Thêm GitHub Secrets

Vào: **Settings → Secrets and variables → Actions → New repository secret**

| Secret | Cách lấy giá trị |
|---|---|
| `CLOUD_CREDENTIALS` | Toàn bộ nội dung `sa-key.json` (copy & paste) |
| `CLOUD_BUCKET` | Tên bucket (ví dụ: `my-mlops-bucket`) |
| `VM_HOST` | IP công khai của VM (từ bước 2.4) |
| `VM_USER` | Tên user trên VM (`echo $USER` khi SSH vào VM) |
| `VM_SSH_KEY` | Toàn bộ nội dung `~/.ssh/mlops_deploy` (private key) |

> Mỗi secret khi paste phải **không có khoảng trắng đầu/cuối**.

### 2.10 Hoàn thiện `tests/test_train.py`

```python
import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = rng.random((n, len(FEATURE_NAMES)))
    y = rng.integers(0, 3, n)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    train_path = str(tmp_path / "train.csv")
    eval_path = str(tmp_path / "eval.csv")
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    return train_path, eval_path


def test_train_returns_float(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    result = train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_metrics_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("outputs/metrics.json")
    with open("outputs/metrics.json") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics
    assert "f1_score" in metrics


def test_model_file_created(tmp_path):
    train_path, eval_path = _make_temp_data(tmp_path)
    train(
        {"n_estimators": 10, "max_depth": 3},
        data_path=train_path,
        eval_path=eval_path,
    )
    assert os.path.exists("models/model.pkl")
```

Chạy test cục bộ trước khi commit:
```bash
pytest tests/ -v
```

Tất cả 3 test phải qua (PASSED).

### 2.11 Hoàn thiện `.github/workflows/mlops.yml`

**Lưu ý quan trọng**: Đổi `branches: [main]` thành `branches: [master]` để khớp với branch hiện tại của repo, hoặc đổi tên branch:
```bash
git branch -m master main
git push origin main
git push origin --delete master
```

Điền các TODO trong workflow:

**TODO 1** - Run tests:
```yaml
- name: Run unit tests
  run: pytest tests/ -v
```

**TODO 2** - Authenticate (GCP):
```yaml
- name: Authenticate to Cloud Storage
  run: |
    echo '${{ secrets.CLOUD_CREDENTIALS }}' > /tmp/sa-key.json
    echo "GOOGLE_APPLICATION_CREDENTIALS=/tmp/sa-key.json" >> $GITHUB_ENV
```

**TODO 3** - DVC pull:
```yaml
- name: Pull data with DVC
  run: dvc pull data/train_phase1.csv.dvc data/eval.csv.dvc
```

**TODO 4** - Read metrics:
```yaml
- name: Read metrics
  id: read_metrics
  run: |
    ACC=$(python -c "import json; d=json.load(open('outputs/metrics.json')); print(d['accuracy'])")
    echo "accuracy=$ACC" >> $GITHUB_OUTPUT
```

**TODO 5** - Upload model (GCP):
```yaml
- name: Upload model to Cloud Storage
  run: |
    python - <<'EOF'
    import os
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(os.environ["CLOUD_BUCKET"])
    blob = bucket.blob("models/latest/model.pkl")
    blob.upload_from_filename("models/model.pkl")
    print("Model uploaded successfully.")
    EOF
  env:
    CLOUD_BUCKET: ${{ secrets.CLOUD_BUCKET }}
```

**TODO 6** - Eval gate:
```yaml
- name: Check eval gate
  run: |
    python - <<'EOF'
    acc = float("${{ needs.train.outputs.accuracy }}")
    if acc < 0.70:
        raise SystemExit(f"FAILED: accuracy {acc:.4f} < 0.70. Hủy deploy.")
    print(f"PASSED: accuracy {acc:.4f} >= 0.70. Đang triển khai...")
    EOF
```

**TODO 7 & 8** - Deploy:
```yaml
- name: SSH deploy to VM
  uses: appleboy/ssh-action@v1.0.3
  with:
    host: ${{ secrets.VM_HOST }}
    username: ${{ secrets.VM_USER }}
    key: ${{ secrets.VM_SSH_KEY }}
    script: |
      sudo systemctl restart mlops-serve
      sleep 5
      curl -sf http://localhost:8000/health && echo "Health check passed." || exit 1
```

### 2.12 Push và chạy pipeline lần đầu

```bash
# Đảm bảo có file __init__.py
touch src/__init__.py tests/__init__.py

git add .
git commit -m "feat: add CI/CD pipeline, tests, and serving API"
git push origin master   # hoặc main nếu đã đổi branch
```

Theo dõi tại tab **Actions** trên GitHub.

Sau khi pipeline chạy thành công, khởi động service trên VM:
```bash
gcloud compute ssh mlops-serve --zone=us-central1-a \
  --command "sudo systemctl start mlops-serve"
```

Test endpoint:
```bash
VM_IP=<YOUR_VM_IP>

curl http://$VM_IP:8000/health
# Kỳ vọng: {"status": "ok"}

curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
# Kỳ vọng: {"prediction": 0, "label": "thấp"}
```

### Checklist Bước 2

- [ ] Bucket trên cloud storage đã được tạo
- [ ] `dvc push` thành công, dữ liệu hiển thị trên Cloud Console
- [ ] 5 GitHub Secrets đã được thêm đúng
- [ ] `pytest tests/ -v` → 3 tests PASSED
- [ ] Cả 4 jobs Actions đều xanh (Unit Test, Train, Eval, Deploy)
- [ ] `curl /health` trả về `{"status": "ok"}`
- [ ] `curl /predict` trả về kết quả dự đoán hợp lệ
- [ ] **Chụp màn hình** 4 jobs màu xanh (bắt buộc nộp bài)

---

## BƯỚC 3 - Continuous Training

### 3.1 Thêm dữ liệu mới

```bash
python add_new_data.py
# Kỳ vọng: Cập nhật dữ liệu: 2998 -> 5996 mẫu
```

Xác nhận:
```bash
# Linux/macOS:
wc -l data/train_phase1.csv   # → 5997 (5996 dòng + header)

# Windows PowerShell:
(Get-Content data/train_phase1.csv).Count
```

### 3.2 Phiên bản hóa và kích hoạt pipeline

**Thứ tự thực hiện RẤT QUAN TRỌNG** (phải `dvc push` trước `git push`):

```bash
# 1. Cập nhật DVC tracking
dvc add data/train_phase1.csv

# 2. Commit file .dvc (KHÔNG phải file CSV)
git add data/train_phase1.csv.dvc
git commit -m "data: bổ sung 2998 mẫu dữ liệu mới (train_phase2)"

# 3. Push dữ liệu lên cloud storage TRƯỚC
dvc push

# 4. Push git commit lên GitHub → kích hoạt Actions
git push origin master   # hoặc main
```

> **Tại sao phải `dvc push` trước `git push`?** Nếu git push trước, GitHub Actions sẽ bắt đầu chạy `dvc pull` nhưng dữ liệu mới chưa có trên cloud → lỗi.

### 3.3 Theo dõi pipeline

- Vào tab **Actions** → thấy pipeline mới được kích hoạt
- Xác nhận commit message trong pipeline là "data: bổ sung 2998 mẫu dữ liệu mới"
- Theo dõi 4 jobs: Unit Test → Train (5996 mẫu) → Eval → Deploy

### 3.4 Xác nhận mô hình mới đã deploy

```bash
VM_IP=<YOUR_VM_IP>
curl http://$VM_IP:8000/health
curl -X POST http://$VM_IP:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
```

### 3.5 So sánh kết quả

Download artifact `metrics.json` từ 2 lần chạy Actions (Bước 2 vs Bước 3) và điền vào bảng:

| Chỉ số | Bước 2 (2998 mẫu) | Bước 3 (5996 mẫu) |
|---|---|---|
| accuracy | ? | ? |
| f1_score | ? | ? |

### Checklist Bước 3

- [ ] Pipeline được kích hoạt tự động bởi commit dữ liệu (không phải commit code)
- [ ] 4 jobs đều xanh
- [ ] `curl /predict` vẫn trả về kết quả đúng
- [ ] Bảng so sánh accuracy đã được điền
- [ ] **Chụp màn hình** Actions tab với commit message là commit dữ liệu

---

## Xử lý sự cố thường gặp

### Branch trigger không khớp

**Vấn đề**: Workflow trigger `branches: [main]` nhưng repo dùng `master`.

**Giải pháp A** - Đổi trigger trong `mlops.yml`:
```yaml
on:
  push:
    branches: [master]
```

**Giải pháp B** - Đổi tên branch Git:
```bash
git branch -m master main
git push origin main
git push origin --delete master
# Vào GitHub Settings → Default branch → đổi sang main
```

### `dvc push` lỗi xác thực

```bash
cat .dvc/config   # kiểm tra credentialpath
export GOOGLE_APPLICATION_CREDENTIALS=sa-key.json
dvc push
```

### GitHub Actions `dvc pull` thất bại

- Kiểm tra secret `CLOUD_CREDENTIALS` chứa đúng nội dung JSON
- Kiểm tra log step "Authenticate to Cloud Storage" xem có lỗi không

### Job Eval thất bại dù accuracy cao

- GitHub Actions outputs là kiểu string, cần `float()` khi so sánh
- Kiểm tra giá trị accuracy được in trong log job Train

### Service VM không khởi động

```bash
gcloud compute ssh mlops-serve --zone=us-central1-a \
  --command "sudo journalctl -u mlops-serve -n 50"
```

Nguyên nhân phổ biến:
- Sai `GCS_BUCKET` trong systemd service
- `sa-key.json` chưa được copy lên VM
- `models/latest/model.pkl` chưa tồn tại trên GCS (service chỉ khởi động sau pipeline đầu tiên)

### Pipeline Bước 3 không được kích hoạt

```bash
git log --name-only -1
# Phải thấy: data/train_phase1.csv.dvc
# Nếu thấy data/train_phase1.csv → đã commit nhầm file CSV
```

---

## Checklist nộp bài

### Ảnh chụp màn hình cần có

1. **MLflow UI** hiển thị ≥ 3 lần chạy với siêu tham số khác nhau
2. **GitHub Actions** tab — 4 jobs màu xanh (Bước 2)
3. **GitHub Actions** tab — 4 jobs màu xanh triggered bởi commit dữ liệu (Bước 3)
4. Output của `curl http://VM_IP:8000/health`
5. Output của `curl http://VM_IP:8000/predict`
6. **Cloud Storage Console** hiển thị file dữ liệu dưới `dvc/` và model dưới `models/latest/model.pkl`

### Nộp bài gồm

1. URL repo GitHub công khai
2. Chuỗi ảnh chụp màn hình theo thứ tự trên
3. File báo cáo ≤ 1 trang A4:
   - Bộ siêu tham số tốt nhất và lý do chọn (từ kết quả Bước 1)
   - Khó khăn gặp phải và cách giải quyết

---

## Rubric điểm

| Hạng mục | Điểm |
|---|---|
| MLflow UI ≥ 3 lần chạy với siêu tham số khác nhau | 12 |
| Mỗi lần chạy có đủ `accuracy` và `f1_score` | 8 |
| Phân tích và chọn siêu tham số tốt nhất | 4 |
| DVC remote đã cấu hình, `dvc push` thành công | 12 |
| 4 GitHub Actions jobs đều xanh (Bước 2) | 16 |
| Deploy bị chặn khi accuracy < 0.70 | 4 |
| VM trả về kết quả đúng tại `/predict` | 12 |
| Một commit dữ liệu kích hoạt toàn bộ pipeline tự động | 12 |
| **Tổng** | **80** |
