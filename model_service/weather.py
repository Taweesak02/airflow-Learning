"""
weather.py — Weather Model Router (Workshop 2, ตัวอย่างขั้นสูง)
================
โมดูลแยกต่างหากสำหรับ endpoint พยากรณ์อุณหภูมิ ใช้โมเดลที่ DAG
ml_02_weather_pipeline.py (Stage 2) เทรนและ deploy ไว้

แยกออกมาจาก main.py เพื่อไม่ให้ไปปนกับโค้ดของโมเดล iris (Stage 1) ที่มีอยู่
เดิม — main.py แค่ import router จากไฟล์นี้ไปใช้ (ดูตัวอย่างท้ายไฟล์นี้)

วิธีต่อเข้ากับ main.py เดิม (ถ้ายังไม่ได้ทำ):
    from weather import router as weather_router
    app.include_router(weather_router)
"""

import os
from datetime import datetime

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Path นี้คือฝั่ง "ในตัว container ของ model_api" — โมเดลอากาศที่ Airflow
# (task deploy_model ในไฟล์ ml_02_weather_pipeline.py) เขียนไว้ที่นี่
WEATHER_MODEL_PATH = "/models/weather_models/current_model.pkl"

router = APIRouter()


class PredictWeatherRequest(BaseModel):
    temp_3_days_ago: float = Field(..., example=28.5)
    temp_2_days_ago: float = Field(..., example=29.0)
    temp_yesterday: float = Field(..., example=29.5)


class PredictWeatherResponse(BaseModel):
    forecast_temperature: float
    model_last_modified: str


def load_weather_model():
    """โหลดโมเดลอากาศล่าสุดจากไฟล์ ทุกครั้งที่เรียก (ไม่ cache) เหมือนโมเดล iris"""
    if not os.path.exists(WEATHER_MODEL_PATH):
        raise HTTPException(
            status_code=503,
            detail=(
                f"ยังไม่มีไฟล์โมเดลที่ {WEATHER_MODEL_PATH} — ต้องรัน DAG "
                "weather_pipeline_dag ใน Airflow อย่างน้อย 1 ครั้งก่อน "
                "(ต้องผ่าน task deploy_model ด้วย)"
            ),
        )
    return joblib.load(WEATHER_MODEL_PATH)


def weather_model_file_info():
    """ใช้โดย /health ใน main.py เพื่อรายงานสถานะไฟล์โมเดลอากาศ"""
    exists = os.path.exists(WEATHER_MODEL_PATH)
    last_modified = None
    if exists:
        last_modified = datetime.fromtimestamp(os.path.getmtime(WEATHER_MODEL_PATH)).isoformat()
    return {"exists": exists, "last_modified": last_modified}


@router.post("/predict_weather", response_model=PredictWeatherResponse)
def predict_weather(payload: PredictWeatherRequest):
    """
    ทำนายอุณหภูมิวันถัดไป จากอุณหภูมิ 3 วันย้อนหลังที่ส่งมา
    ใช้ feature เดียวกับตอนเทรนใน ml_02_weather_pipeline.py:
    [lag1, lag2, lag3, moving_average] โดย lag1 คือ "เมื่อวาน" (ใกล้ที่สุด)
    """
    model = load_weather_model()

    lag1 = payload.temp_yesterday
    lag2 = payload.temp_2_days_ago
    lag3 = payload.temp_3_days_ago
    moving_avg = (lag1 + lag2 + lag3) / 3

    features = np.array([[lag1, lag2, lag3, moving_avg]])
    forecast = float(model.predict(features)[0])

    last_modified = datetime.fromtimestamp(os.path.getmtime(WEATHER_MODEL_PATH)).isoformat()

    return PredictWeatherResponse(
        forecast_temperature=round(forecast, 2),
        model_last_modified=last_modified,
    )


# หน้าฟอร์มทดสอบ (HTML fragment) — main.py ดึงไปแปะต่อท้ายหน้าเว็บ iris เดิม
WEATHER_FORM_HTML = """
  <hr style="margin: 32px 0; border: none; border-top: 1px solid #e5e7eb;">

  <h1>🌦️ Weather Model — ทำนายอุณหภูมิพรุ่งนี้</h1>
  <p style="color:#6b7280; font-size:14px;">กรอกอุณหภูมิ 3 วันย้อนหลัง (°C) แล้วกด "พยากรณ์" เพื่อเรียกโมเดลล่าสุดที่ Airflow deploy ไว้</p>

  <div class="field">
    <label>อุณหภูมิเมื่อ 3 วันก่อน (°C)</label>
    <input type="number" step="0.1" id="temp_3_days_ago" value="28.5">
  </div>
  <div class="field">
    <label>อุณหภูมิเมื่อ 2 วันก่อน (°C)</label>
    <input type="number" step="0.1" id="temp_2_days_ago" value="29.0">
  </div>
  <div class="field">
    <label>อุณหภูมิเมื่อวาน (°C)</label>
    <input type="number" step="0.1" id="temp_yesterday" value="29.5">
  </div>

  <button onclick="doPredictWeather()">พยากรณ์</button>

  <div id="weather_result"></div>

<script>
async function doPredictWeather() {
  const payload = {
    temp_3_days_ago: parseFloat(document.getElementById('temp_3_days_ago').value),
    temp_2_days_ago: parseFloat(document.getElementById('temp_2_days_ago').value),
    temp_yesterday: parseFloat(document.getElementById('temp_yesterday').value),
  };

  const resultDiv = document.getElementById('weather_result');
  resultDiv.style.display = 'block';
  resultDiv.className = '';
  resultDiv.innerHTML = 'กำลังพยากรณ์...';

  try {
    const res = await fetch('/predict_weather', {
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
      <div style="font-size:18px; font-weight:bold;">🌡️ พยากรณ์พรุ่งนี้: ${data.forecast_temperature} °C</div>
      <div class="meta">โมเดลอัปเดตล่าสุด: ${data.model_last_modified}</div>
    `;
  } catch (e) {
    resultDiv.className = 'err';
    resultDiv.innerHTML = '❌ เรียก API ไม่สำเร็จ: ' + e.message;
  }
}
</script>
"""