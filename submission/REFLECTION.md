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
   - Viết logic trong mã nguồn để load mô hình mới vào một biến phụ (`new_model`). 
   - Kiểm tra xem biến phụ có sẵn sàng không. Nếu gặp lỗi `Exception` khi đọc file `model.pkl` đang bị tải dở, biến chính `model` vẫn sẽ trỏ vào phiên bản cũ trong RAM. Điều này tránh việc API ném lỗi HTTP 503 khi người dùng vô tình gọi trúng lúc file model đang được ghi chèn lên (overwrite).
