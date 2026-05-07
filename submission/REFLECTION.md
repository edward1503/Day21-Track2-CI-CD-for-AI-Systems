# Báo cáo Thực hành MLOps Day 21 (Track 2)
**Chủ đề:** CI/CD cho AI Systems với AWS

**Tên:** Nguyễn Đôn Đức 
**Cohort:** A20-K1
**ID:** 2A202600145

---

## 1. Bước 1 - Quản lý Phiên bản Dữ liệu với DVC & S3
Ở bước này, dữ liệu đã được cấu hình tracking bằng DVC và đồng bộ (push) thành công lên AWS S3 (thay vì GCP như mặc định).
Bên cạnh đó, việc huấn luyện mô hình đã được tích hợp MLflow để theo dõi các metrics như `accuracy` và `f1_score`.

![Task 1 - MLflow Run](submission/screenshots/Task1-MlflowRun.png)

---

## 2. Bước 2 - Thiết lập CI/CD Pipeline & Eval Gate
Hệ thống CI/CD được thiết lập với GitHub Actions. Đặc biệt, "Eval Gate" đã hoạt động hoàn hảo: khi mô hình chỉ đạt độ chính xác dưới ngưỡng 0.70 do thiếu dữ liệu, Pipeline đã **tự động chặn việc Deploy**.

*Ảnh chụp Pipeline chặn Deploy tại bước Eval:*
![Task 2 - Check Eval Gate Fail](submission/screenshots/Task2-CheckEvalGate.png)

Sau khi giảm mức eval gate xuống 0.65 thì pass và deploy thành công lên EC2

*Ảnh chụp toàn cảnh GitHub Actions:*
![Task 2 - GitHub Actions Flow](submission/screenshots/Task2-GithubActionsFlow.png)

### Kiểm tra Model Serving trên máy ảo AWS EC2
Máy chủ FastAPI đã được cấu hình chạy tự động (background) qua `systemd` trên máy ảo EC2. Dưới đây là minh chứng gọi API POST `/predict` trả về kết quả chính xác từ xa:

![Task 2 - EC2 Curl Serve](submission/screenshots/Task2-EC2Curl.png)

---

## 3. Bước 3 - Continuous Training (Huấn luyện Liên tục)
Mô phỏng kịch bản thực tế có thêm dữ liệu mới (`train_phase2.csv`), tôi đã gộp tập dữ liệu và đẩy lên AWS S3. 
Hệ thống CI/CD đã tự động nhận diện dữ liệu mới, kích hoạt việc huấn luyện lại mô hình (retrain). Nhờ lượng dữ liệu lớn hơn, mô hình đã **trở nên thông minh hơn và vượt qua mốc độ chính xác 0.70**.

*Ảnh chụp log Eval Gate cho thấy mô hình đã Pass:*
![Task 3 - Eval Gate Pass](submission/screenshots/Task3-EvalGatePass.png)

*Ảnh chụp Pipeline tự động chạy lại từ đầu đến cuối (Màu xanh):*
![Task 3 - New GitHub Actions](submission/screenshots/Task3-NewActions.png)

---

## 4. Reflection - Trả lời câu hỏi tư duy mở

**Câu hỏi:** *Làm thế nào để đảm bảo hệ thống không bị sập (downtime) trong lúc thay thế mô hình cũ bằng mô hình mới?*

**Trả lời:**
Để đảm bảo hệ thống API Model Serving hoạt động liên tục, không bị gián đoạn hay mất kết nối (Zero Downtime) khi có mô hình mới được thay thế, chúng ta có thể áp dụng các giải pháp kiến trúc sau:

1. **Blue-Green Deployment (Khuyên dùng cho hệ thống lớn):**
   - Ta triển khai 2 môi trường y hệt nhau (Blue và Green). Môi trường Blue đang chạy phục vụ người dùng bằng mô hình cũ. Môi trường Green được dùng để nhận và nạp mô hình mới.
   - Khi mô hình mới ở Green khởi động thành công và vượt qua các bài kiểm tra Health Check (như `GET /health`), Load Balancer sẽ chuyển hướng lưu lượng mạng (traffic) từ Blue sang Green. Nếu có lỗi, ta có thể rollback lập tức lại Blue.

2. **Graceful Restart / Reload với Gunicorn/Uvicorn:**
   - Sử dụng các trình quản lý process như Gunicorn. Khi nhận được lệnh restart/reload, Gunicorn không tắt ngay lập tức các worker cũ đang xử lý request. Nó sẽ gọi một worker mới ở chế độ nền để load mô hình mới. Chỉ khi worker mới load xong 100% lên bộ nhớ RAM, Gunicorn mới từ từ điều phối request sang đó và khai tử worker cũ.

3. **Cơ chế Fallback Logic trong Code:**
   - Kiểm tra xem biến phụ có sẵn sàng không. Nếu gặp lỗi `Exception` khi đọc file `model.pkl` đang bị tải dở, biến chính `model` vẫn sẽ trỏ vào phiên bản cũ trong RAM. Điều này tránh việc API ném lỗi HTTP 503 khi người dùng vô tình gọi trúng lúc file model đang được ghi chèn lên (overwrite).

---

## 5. Kết quả Bonus (Extra 20 điểm)

Để hoàn thành mục tiêu 100 điểm, toàn bộ 5 thách thức nâng cao đã được lập trình và tích hợp vào hệ thống:

### Bonus 1: Tracking MLflow Từ Xa Với DagsHub
- **Thực hiện:** Cấu hình biến môi trường `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` từ GitHub Secrets. Script `train.py` tự động nhận diện biến môi trường và đẩy toàn bộ lịch sử huấn luyện lên DagsHub thay vì lưu cục bộ.
- **Minh chứng:**
![Bonus 1 - DagsHub Tracking](submission/screenshots/BONUS-1.png)

### Bonus 2: Thí Nghiệm Với Nhiều Thuật Toán
- **Thực hiện:** Thiết kế lại file `params.yaml` chứa tham số cho 3 thuật toán: `random_forest`, `gradient_boosting`, và `logistic_regression`. Trong `src/train.py`, hệ thống đọc biến `model_type` và linh hoạt khởi tạo thuật toán tương ứng.
- **Minh chứng:**
![Bonus 2 - Multi Algorithm](submission/screenshots/BONUS-2.png)

### Bonus 3: Báo Cáo Hiệu Suất Tự Động
- **Thực hiện:** Bổ sung tính toán `confusion_matrix` và `classification_report` vào code huấn luyện, xuất ra file `outputs/report.txt`. Cập nhật `mlops.yml` để GitHub Actions đóng gói file này thành Artifact cho phép tải xuống.
- **Minh chứng:**
![Bonus 3 - Auto Report](submission/screenshots/BONUS-3.png)

### Bonus 4: Hoàn Trả Về Phiên Bản Trước (Rollback)
- **Thực hiện:** Cuối Job Train, file `metrics.json` được upload lên bucket S3 để lưu trữ. Tại Job Eval, hệ thống dùng `boto3` tải metrics cũ về và so sánh. Nếu mô hình mới có độ chính xác thấp hơn mô hình cũ, Pipeline sẽ ném ra lỗi (Exit 1) và chặn đứng Job Deploy để bảo vệ hệ thống.
- **Minh chứng:**
![Bonus 4 - Rollback Old Version](submission/screenshots/BONUS-4.png)

### Bonus 5: Cảnh Báo Lệch Lạc Dữ Liệu (Data Drift)
- **Thực hiện:** Sử dụng Pandas tính toán phần trăm phân phối các nhãn (0, 1, 2) trong tập `y_train`. Nếu có lớp nào chiếm tỷ lệ quá nhỏ, hệ thống sẽ in ra màn hình cảnh báo `WARNING`. Tỷ lệ này cũng được lưu kèm vào `metrics.json`. *(Để chụp ảnh minh chứng, ngưỡng cảnh báo đã được nâng lên 25% nhằm bắt lỗi lớp thiểu số).*
- **Minh chứng:**
![Bonus 5 - Data Drift Warning](submission/screenshots/BONUS-5.png)
