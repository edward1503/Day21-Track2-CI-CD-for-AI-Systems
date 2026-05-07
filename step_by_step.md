# Hướng Dẫn Thực Hiện Từng Bước (Step-by-Step) - Lab CI/CD cho AI Systems

Dưới đây là các bước hành động chi tiết để hoàn thành bài lab. Hãy kết hợp tài liệu này cùng file `PLAN.md` (nơi chứa code chi tiết) để thực hành.

---

## 1. Chuẩn Bị Môi Trường (Thực hiện một lần)
1. **Clone repo**, tạo và kích hoạt môi trường ảo (virtualenv):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux/macOS
   source .venv/bin/activate
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Chạy script tạo dữ liệu ban đầu:
   ```bash
   python generate_data.py
   ```

---

## 2. Phase 1: Thực Nghiệm Cục Bộ (Local Experimentation)
1. Cấu hình biến môi trường cục bộ để lưu dữ liệu MLflow (nhớ tạo file `.env` hoặc gán thủ công):
   ```powershell
   $env:MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
   $env:MLFLOW_ARTIFACT_ROOT = "./mlartifacts"
   ```
2. Hoàn thiện script `src/train.py` (sao chép đoạn code hoàn chỉnh từ mục **1.2** trong `PLAN.md`).
3. Chạy lệnh `python src/train.py` ít nhất **3 lần** với các tham số khác nhau:
   - Sửa file `params.yaml` trước mỗi lần chạy để thay đổi các siêu tham số (hyperparameters).
4. Khởi chạy giao diện MLflow để xem kết quả:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
5. Đánh giá và chọn bộ tham số tốt nhất, cập nhật cố định lại vào file `params.yaml`. **(Nhớ chụp hình UI này để nộp bài).**

---

## 3. Phase 2: CI/CD Pipeline Tự Động (Automated Pipeline - AWS)
1. **Thiết lập S3 Bucket & DVC:**
   - Tạo S3 Bucket trên AWS (VD: lệnh `aws s3 mb s3://<TEN_BUCKET_CUA_BAN> --region us-east-1`).
   - Tạo IAM User với quyền truy cập vào S3 (VD: `AmazonS3FullAccess`) và tạo Access Key để lấy `AWS_ACCESS_KEY_ID` cùng `AWS_SECRET_ACCESS_KEY`.
   - Cấu hình thông tin xác thực AWS cục bộ trên máy bạn bằng lệnh `aws configure` (hoặc thiết lập biến môi trường).
   - Khởi tạo DVC và liên kết với remote S3 bucket:
     ```bash
     dvc init
     dvc remote add -d myremote s3://<TEN_BUCKET_CUA_BAN>/dvc
     ```
   - Dùng DVC track các file dữ liệu `.csv`:
     ```bash
     dvc add data/train_phase1.csv
     dvc add data/eval.csv
     dvc add data/train_phase2.csv
     ```
   - Commit file tracking `.dvc` (không commit file `.csv`) và cấu hình lên Git.
   - Đẩy dữ liệu thật lên S3: `dvc push`.
2. **Khởi Tạo và Cấu Hình EC2 VM:**
   - Khởi tạo một máy ảo EC2 (chọn hệ điều hành Ubuntu) trên AWS, cấu hình Security Group để mở port **8000** (cho API) và **22** (để SSH). Lấy Public IP và tải file `.pem` về.
   - SSH vào máy ảo EC2, cài đặt Python (`sudo apt update && sudo apt install -y python3-pip`), cài các thư viện `fastapi`, `uvicorn`, `scikit-learn`, `joblib`, `boto3`.
   - Thiết lập thông tin xác thực AWS trên máy ảo bằng cách cấu hình qua lệnh `aws configure` hoặc thêm khóa IAM vào các biến môi trường của hệ thống.
3. **Cấu hình Web API (`src/serve.py`):**
   - Tạo file `src/serve.py` chứa API phục vụ model. Chú ý: Cần chỉnh sửa hàm `download_model` để dùng thư viện `boto3` tải model từ S3 (thay vì `google-cloud-storage` như trong PLAN.md).
   - Copy `src/serve.py` lên máy ảo và cài đặt Service chạy ngầm (systemd) theo form hướng dẫn ở mục **2.7** của `PLAN.md`, đổi các tham chiếu chứng chỉ GCP sang các biến môi trường cấu hình của AWS.
4. **Thiết Lập GitHub Actions:**
   - Đăng nhập vào GitHub repo, thêm các Secret: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `CLOUD_BUCKET` (Tên bucket S3), `VM_HOST` (IP EC2), `VM_USER` (thường là `ubuntu`), `VM_SSH_KEY` (Nội dung của file `.pem` tải từ EC2).
   - Cập nhật script kiểm thử và file cấu hình CI/CD `.github/workflows/mlops.yml` (chú ý sửa các step GCP sang AWS như dùng aws-actions/configure-aws-credentials để cấu hình chứng chỉ AWS, và step upload file thay bằng lệnh aws s3 cp hoặc python với boto3).
   - Push code (Gồm file `.dvc`, `.github`, `src/`) lên GitHub.
   - Mở tab **Actions** trên GitHub, chờ Pipeline chạy thành công cả 4 jobs. **(Chụp màn hình luồng chạy này).**
5. **Kiểm Tra Thực Tế:**
   - Start service ngầm trên máy ảo: `sudo systemctl start mlops-serve`.
   - Gọi test 2 API từ máy local qua IP EC2:
     ```bash
     curl http://<VM_IP>:8000/health
     curl -X POST http://<VM_IP>:8000/predict -H "Content-Type: application/json" -d '{"features": [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}'
     ```
   - **(Chụp màn hình kết quả lệnh Curl).**

---

## 4. Phase 3: Continuous Training (Huấn Luyện Liên Tục)
1. Sinh thêm dữ liệu (giả lập có dữ liệu mới tới):
   ```bash
   python add_new_data.py
   ```
2. Track lại file dữ liệu mới bằng DVC:
   ```bash
   dvc add data/train_phase1.csv
   ```
3. **Bước Rất Quan Trọng:** Push dữ liệu thật lên Cloud Storage **TRƯỚC**:
   ```bash
   dvc push
   ```
4. Commit sự thay đổi file DVC tracking (thể hiện việc dữ liệu đã thay đổi phiên bản) và đẩy lên GitHub:
   ```bash
   git add data/train_phase1.csv.dvc
   git commit -m "data: cập nhật dữ liệu huấn luyện phase 2"
   git push origin <tên-branch>
   ```
5. Truy cập lại tab **Actions** trên GitHub, quan sát Pipeline tự động được kích hoạt và chạy qua 4 quá trình tương tự Phase 2. **(Chụp màn hình sự kiện này).**
6. Thử gọi lại API một lần nữa qua lệnh `curl` để xác nhận server API vẫn hoạt động tốt trên model mới.
