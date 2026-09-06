"""Serve the evaluated HeartDisease pipeline produced by Airflow."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()
MODEL_DIR = Path("/models/heart_failure_models")
# DSN ของ postgres_target มาจาก docker-compose; ไม่ตั้งไว้ = ปิดการเก็บ log
PREDICTION_LOG_DSN = os.getenv("PREDICTION_LOG_DSN", "")


class PredictHeartRequest(BaseModel):
    Age: int = Field(..., gt=0)
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"]
    RestingBP: float = Field(..., ge=0, allow_inf_nan=False)
    Cholesterol: float = Field(..., ge=0, allow_inf_nan=False)
    FastingBS: Literal[0, 1]
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(..., gt=0, allow_inf_nan=False)
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(..., allow_inf_nan=False)
    ST_Slope: Literal["Up", "Flat", "Down"]


class PredictHeartResponse(BaseModel):
    predicted_HeartDisease: int
    probability_HeartDisease: float
    model_run: str
    model_last_modified: str


def latest_model_path():
    # Only serve runs that finished evaluation, avoiding a model still being written.
    candidates = [p for p in MODEL_DIR.glob("*/model.joblib")
                  if (p.parent / "metrics.json").is_file()]
    return max(candidates, key=lambda p: p.stat().st_mtime_ns, default=None)


def log_prediction(model_run, prediction, probability, features):
    """เก็บทุก request ลง heart_prediction_log เพื่อให้ย้อนดูได้ว่าโมเดลตอบอะไรไปบ้าง

    ห้าม raise เด็ดขาด — การเก็บ log ล้มเหลวต้องไม่ทำให้คนไข้ไม่ได้ผลทำนาย
    ตาราง heart_prediction_log สร้างโดย task create_tables ของ DAG
    """
    if not PREDICTION_LOG_DSN:
        return
    try:
        with psycopg2.connect(PREDICTION_LOG_DSN, connect_timeout=3) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO heart_prediction_log "
                    "(model_run, predicted_heartdisease, probability_heartdisease, features) "
                    "VALUES (%s, %s, %s, %s);",
                    (model_run, prediction, probability, Json(features)),
                )
    except Exception as exc:  # noqa: BLE001 - log ล้มเหลวต้องไม่กระทบการทำนาย
        print(f"[prediction_log] บันทึกไม่สำเร็จ ({type(exc).__name__}: {exc})")


def heart_model_file_info():
    path = latest_model_path()
    return {
        "exists": path is not None,
        "last_modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path else None,
        "model_run": path.parent.name if path else None,
    }


@router.post("/predict_heart_failure", response_model=PredictHeartResponse)
def predict_heart_failure(payload: PredictHeartRequest):
    path = latest_model_path()
    if path is None:
        raise HTTPException(503, "ยังไม่มีโมเดล กรุณารัน DAG heart_failure_pipeline_dag ให้ผ่าน evaluate_model ก่อน")
    model = joblib.load(path)
    payload_dict = payload.model_dump()
    features = pd.DataFrame([payload_dict])
    # Match validate_features in the training DAG before pipeline preprocessing.
    features[["RestingBP", "Cholesterol"]] = features[["RestingBP", "Cholesterol"]].replace(0, np.nan)
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, list(model.classes_).index(1)])
    log_prediction(path.parent.name, prediction, probability, payload_dict)
    return PredictHeartResponse(
        predicted_HeartDisease=prediction,
        probability_HeartDisease=probability,
        model_run=path.parent.name,
        model_last_modified=datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
    )


HEART_FORM_HTML = """
  <hr style="margin:32px 0; border:none; border-top:1px solid #e5e7eb;">
  <h1>❤️ Heart Failure Model — ทำนายโรคหัวใจ</h1>
  <p class="meta">กรอกข้อมูล 11 ค่าเพื่อทำนาย HeartDisease จากโมเดลล่าสุดของ Airflow</p>
  <form id="heart_form">
    <div class="field"><label for="heart_Age">อายุ (ปี)</label><input id="heart_Age" name="Age" type="number" min="1" step="1" value="40" required></div>
    <div class="field"><label for="heart_Sex">เพศ (Sex)</label><select id="heart_Sex" name="Sex"><option value="M">ชาย (M)</option><option value="F">หญิง (F)</option></select></div>
    <div class="field"><label for="heart_ChestPainType">ลักษณะอาการเจ็บหน้าอก</label><select id="heart_ChestPainType" name="ChestPainType"><option value="ATA">Atypical angina (ATA)</option><option value="TA">Typical angina (TA)</option><option value="NAP">Non-anginal pain (NAP)</option><option value="ASY">Asymptomatic (ASY)</option></select></div>
    <div class="field"><label for="heart_RestingBP">ความดันขณะพัก (mmHg)</label><input id="heart_RestingBP" name="RestingBP" type="number" min="0" step="any" value="140" required></div>
    <div class="field"><label for="heart_Cholesterol">คอเลสเตอรอล (mg/dL)</label><input id="heart_Cholesterol" name="Cholesterol" type="number" min="0" step="any" value="289" required></div>
    <p class="meta">ค่า 0 ของความดันและคอเลสเตอรอลจะใช้ค่าทดแทนจากข้อมูลฝึก</p>
    <div class="field"><label for="heart_FastingBS">น้ำตาลขณะอดอาหาร (FastingBS)</label><select id="heart_FastingBS" name="FastingBS"><option value="0">ไม่เกิน 120 mg/dL (0)</option><option value="1">มากกว่า 120 mg/dL (1)</option></select></div>
    <div class="field"><label for="heart_RestingECG">ผล ECG ขณะพัก</label><select id="heart_RestingECG" name="RestingECG"><option value="Normal">Normal</option><option value="ST">ST</option><option value="LVH">LVH</option></select></div>
    <div class="field"><label for="heart_MaxHR">อัตราการเต้นหัวใจสูงสุด (ครั้ง/นาที)</label><input id="heart_MaxHR" name="MaxHR" type="number" min="1" step="any" value="172" required></div>
    <div class="field"><label for="heart_ExerciseAngina">เจ็บหน้าอกเมื่อออกกำลังกาย</label><select id="heart_ExerciseAngina" name="ExerciseAngina"><option value="N">ไม่ (N)</option><option value="Y">ใช่ (Y)</option></select></div>
    <div class="field"><label for="heart_Oldpeak">ST depression (Oldpeak)</label><input id="heart_Oldpeak" name="Oldpeak" type="number" step="any" value="0" required></div>
    <div class="field"><label for="heart_ST_Slope">ความชัน ST (ST_Slope)</label><select id="heart_ST_Slope" name="ST_Slope"><option value="Up">Up</option><option value="Flat">Flat</option><option value="Down">Down</option></select></div>
    <button id="heart_submit" type="submit">ทำนายโรคหัวใจ</button>
  </form>
  <div id="heart_result" role="status" aria-live="polite" style="white-space:pre-line"></div>
  <p class="meta">ผลนี้เป็นการทำนาย HeartDisease ของโมเดลเพื่อการศึกษา ไม่ใช่การวินิจฉัยภาวะหัวใจล้มเหลว</p>
<script>
document.getElementById('heart_form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = Object.fromEntries(new FormData(form));
  for (const key of ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']) {
    payload[key] = Number(payload[key]);
  }
  const result = document.getElementById('heart_result');
  const button = document.getElementById('heart_submit');
  button.disabled = true;
  result.style.display = 'block';
  result.className = '';
  result.textContent = 'กำลังทำนาย...';
  try {
    const response = await fetch('/predict_heart_failure', {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail)
        ? data.detail.map(item => item.loc.join('.') + ': ' + item.msg).join('; ')
        : data.detail;
      throw new Error(detail || 'เกิดข้อผิดพลาดในการทำนาย');
    }
    result.className = 'ok';
    result.textContent = `ผลทำนาย: ${data.predicted_HeartDisease === 1 ? 'โรคหัวใจ (1)' : 'ปกติ (0)'}
ความน่าจะเป็นของ HeartDisease = 1 จากโมเดล: ${(data.probability_HeartDisease * 100).toFixed(1)}%
โมเดลอัปเดตล่าสุด: ${data.model_last_modified}
Run: ${data.model_run}`;
  } catch (error) {
    result.className = 'err';
    result.textContent = 'เรียก API ไม่สำเร็จ: ' + error.message;
  } finally {
    button.disabled = false;
  }
});
</script>
"""
