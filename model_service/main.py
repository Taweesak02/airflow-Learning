"""
main.py — Iris, Weather and Heart Failure Model Serving API
================
FastAPI service แยกจาก Airflow โดยสิ้นเชิง ทำหน้าที่แค่อย่างเดียวคือ
"เสิร์ฟโมเดลล่าสุดที่ Airflow deploy ไว้" ให้เรียกใช้งานผ่าน HTTP ได้

ไฟล์นี้ดูแลเฉพาะโมเดล iris (Stage 1) เท่านั้น ส่วนโมเดลพยากรณ์อากาศ
(Stage 2) แยกออกไปอยู่ที่ weather.py ต่างหาก แล้ว import เข้ามาต่อ
(ผ่าน app.include_router) เพื่อไม่ให้โค้ดสองส่วนนี้ปนกัน
Heart Failure แยกไว้ที่ heart_failure.py และใช้โมเดลจาก run ล่าสุดที่ประเมินผลแล้ว

แนวคิดสำคัญ:
- ไม่ cache โมเดลไว้ในหน่วยความจำ โหลดจากไฟล์ใหม่ทุกครั้งที่มีการเรียก
  /predict ทำให้ได้โมเดลล่าสุดเสมอโดยไม่ต้อง restart service หรือทำ
  endpoint /reload แยกต่างหาก (แลกกับ performance เล็กน้อย ซึ่งรับได้
  สำหรับ demo/งานที่ traffic ไม่สูงมาก)
- อ่านไฟล์โมเดลจาก path เดียวกับที่ DAG deploy_model เขียนไว้ ผ่าน
  Docker volume ที่ mount ร่วมกันระหว่าง Airflow container กับ service นี้
  (ดู docker-compose.yaml ส่วน model_api และ x-airflow-common)

รันแยกจาก Airflow อย่างสิ้นเชิง คนละ container คนละ process
"""

import os
import time
from datetime import datetime

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from sklearn.datasets import load_iris

from weather import router as weather_router, weather_model_file_info, WEATHER_FORM_HTML
from heart_failure import router as heart_router, heart_model_file_info, HEART_FORM_HTML

# Path นี้คือฝั่ง "ในตัว container ของ model_api" (ดู docker-compose.yaml
# ที่ mount โฟลเดอร์เดียวกับที่ Airflow เขียนโมเดลไว้ มาที่ /models)
MODEL_PATH = "/models/iris_models/current_model.pkl"

TARGET_NAMES = load_iris().target_names.tolist()

app = FastAPI(title="Iris + Weather + Heart Failure Model Serving API", version="1.2.0")
app.include_router(weather_router)   # เพิ่ม endpoint /predict_weather จากไฟล์ weather.py
app.include_router(heart_router)


class PredictRequest(BaseModel):
    sepal_length: float = Field(..., example=5.1)
    sepal_width: float = Field(..., example=3.5)
    petal_length: float = Field(..., example=1.4)
    petal_width: float = Field(..., example=0.2)


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_last_modified: str


def load_current_model():
    """โหลดโมเดล iris ล่าสุดจากไฟล์ ทุกครั้งที่เรียก (ไม่ cache)"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                "ยังไม่มีไฟล์โมเดล — ต้องรัน DAG iris_pipeline_dag ใน Airflow "
                "อย่างน้อย 1 ครั้งก่อน (ต้องผ่าน task deploy_model ด้วย)"
            ),
        )
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    """เช็คสถานะ service และข้อมูลโมเดล Iris, Weather, Heart Failure"""
    exists = os.path.exists(MODEL_PATH)
    last_modified = None
    if exists:
        last_modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()

    return {
        "status": "ok",
        "iris_model": {"exists": exists, "last_modified": last_modified},
        "weather_model": weather_model_file_info(),   # มาจาก weather.py
        "heart_failure_model": heart_model_file_info(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    """ทำนายพันธุ์ดอกไม้จาก 4 ค่าที่ส่งมา ใช้โมเดลล่าสุดที่ Airflow deploy ไว้"""
    model = load_current_model()

    features = np.array([[
        payload.sepal_length,
        payload.sepal_width,
        payload.petal_length,
        payload.petal_width,
    ]])

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[prediction])

    last_modified = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH)).isoformat()

    return PredictResponse(
        predicted_class=TARGET_NAMES[prediction],
        confidence=round(confidence, 4),
        model_last_modified=last_modified,
    )


@app.get("/", response_class=HTMLResponse)
def test_page():
    """หน้าเว็บทดสอบง่ายๆ ในตัว ไม่ต้องมี frontend แยก กรอกค่าแล้วกดทายได้เลย"""

    htmlSetup = """
<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Iris + Weather + Heart Failure — หน้าทดสอบ</title>
<style>
  body { font-family: 'TH Sarabun New', Tahoma, sans-serif; max-width: 480px; margin: 40px auto; padding: 0 16px; color: #1f2937; }
  h1 { font-size: 22px; color: #1F4E79; }
  .field { margin-bottom: 14px; }
  label { display: block; font-size: 15px; margin-bottom: 4px; color: #374151; }
  input, select { width: 100%; padding: 8px 10px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 15px; box-sizing: border-box; }
  button:disabled { opacity: 0.6; cursor: wait; }
  #heart_result { margin-top: 20px; padding: 14px; border-radius: 8px; display: none; }
  #heart_result.ok { background: #E1F5EE; border-left: 4px solid #085041; }
  #heart_result.err { background: #FCEBEB; border-left: 4px solid #791F1F; }
  button { width: 100%; padding: 10px; background: #1F4E79; color: white; border: none; border-radius: 6px; font-size: 16px; cursor: pointer; margin-top: 8px; }
  button:hover { background: #2E74B5; }
  #result, #weather_result { margin-top: 20px; padding: 14px; border-radius: 8px; display: none; }
  #result.ok, #weather_result.ok { background: #E1F5EE; border-left: 4px solid #085041; }
  #result.err, #weather_result.err { background: #FCEBEB; border-left: 4px solid #791F1F; }
  .meta { font-size: 13px; color: #6b7280; margin-top: 6px; }
</style>
</head>
<body>
"""

    irisWeb = """
  <h1>🌸 Iris Model — หน้าทดสอบ</h1>
  <p style="color:#6b7280; font-size:14px;">กรอกค่าดอกไม้ 4 ค่า แล้วกด "ทำนาย" เพื่อเรียกโมเดลล่าสุดที่ Airflow deploy ไว้</p>

  <div class="field">
    <label>Sepal length (cm)</label>
    <input type="number" step="0.1" id="sepal_length" value="5.1">
  </div>
  <div class="field">
    <label>Sepal width (cm)</label>
    <input type="number" step="0.1" id="sepal_width" value="3.5">
  </div>
  <div class="field">
    <label>Petal length (cm)</label>
    <input type="number" step="0.1" id="petal_length" value="1.4">
  </div>
  <div class="field">
    <label>Petal width (cm)</label>
    <input type="number" step="0.1" id="petal_width" value="0.2">
  </div>

  <button onclick="doPredict()">ทำนาย</button>

  <div id="result"></div>
"""

    scriptSetup = """
<script>
async function doPredict() {
  const payload = {
    sepal_length: parseFloat(document.getElementById('sepal_length').value),
    sepal_width: parseFloat(document.getElementById('sepal_width').value),
    petal_length: parseFloat(document.getElementById('petal_length').value),
    petal_width: parseFloat(document.getElementById('petal_width').value),
  };

  const resultDiv = document.getElementById('result');
  resultDiv.style.display = 'block';
  resultDiv.className = '';
  resultDiv.innerHTML = 'กำลังทำนาย...';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      resultDiv.className = 'err';
      resultDiv.innerHTML = '❌ ' + (data.detail || 'เกิดข้อผิดพลาด');
      return;
    }

    resultDiv.className = 'ok';
    resultDiv.innerHTML = `
      <div style="font-size:18px; font-weight:bold;">🌼 ผลทำนาย: ${data.predicted_class}</div>
      <div>ความมั่นใจ: ${(data.confidence * 100).toFixed(1)}%</div>
      <div class="meta">โมเดลอัปเดตล่าสุด: ${data.model_last_modified}</div>
    `;
  } catch (e) {
    resultDiv.className = 'err';
    resultDiv.innerHTML = '❌ เรียก API ไม่สำเร็จ: ' + e.message;
  }
}
  </script>
  """
    # แปะฟอร์มพยากรณ์อากาศ (มาจาก weather.py) ต่อท้ายก่อนปิด </body></html>
    # return htmlSetup + irisWeb + WEATHER_FORM_HTML + scriptSetup + "</body>\n</html>\n"
    return htmlSetup + HEART_FORM_HTML + scriptSetup + "</body>\n</html>\n"
 
    
