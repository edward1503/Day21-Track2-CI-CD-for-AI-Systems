from fastapi import FastAPI, HTTPException
import boto3
import joblib
import os

app = FastAPI()

S3_BUCKET = os.environ["CLOUD_BUCKET"] # Lấy từ biến môi trường
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")

def download_model():
    s3_client = boto3.client('s3')
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    s3_client.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    print(f"Model downloaded to {MODEL_PATH}")

try:
    download_model()
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Lưu ý: Model chưa tồn tại hoặc lỗi tải xuống. Chờ pipeline chạy lần đầu. Lỗi: {e}")
    model = None

from pydantic import BaseModel

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
         raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail="Expected 12 features (wine quality)")
    pred = model.predict([req.features])[0]
    labels = {0: "thấp", 1: "trung_bình", 2: "cao"}
    return {"prediction": int(pred), "label": labels[int(pred)]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
