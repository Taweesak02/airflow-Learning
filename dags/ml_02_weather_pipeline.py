"""
ml_02_weather_pipeline.py
================
Workshop 2: ML Pipeline บน Apache Airflow (Stage 2 - Open-Meteo regression)
*** ไฟล์หลักของ Workshop 2 ***

เป้าหมายของไฟล์นี้:
- เทรนโมเดลทำนาย "อุณหภูมิของวันพรุ่งนี้" จากข้อมูลย้อนหลัง 3 วันก่อนหน้า
  (regression) ของจังหวัดเดียว (ไม่ทำหลายจังหวัดเหมือน workshop 1 stage 4
  เพื่อโฟกัสที่แนวคิดใหม่ของไฟล์นี้แทน)
- ยกระดับจาก Stage 1: เกณฑ์ deploy ไม่ใช่ค่าคงที่แล้ว แต่ต้อง "ดีกว่าโมเดล
  รอบก่อนหน้า" (champion-challenger) โดยเก็บ/ดึงค่า RMSE รอบก่อนจาก
  Postgres โดยตรง (ไม่ใช้ Airflow Variables) ใช้ connection postgres_target
  ตัวเดียวกับที่ workshop 1 ตั้งไว้แล้ว ไม่ต้องตั้งค่าใหม่
- Bootstrap ข้อมูลย้อนหลังอัตโนมัติเป็น task แรกในตัว DAG เอง แบบ idempotent
  (เช็คจำนวนแถวที่มีอยู่ก่อน ถ้าไม่พอค่อยดึงเพิ่มจาก Open-Meteo Historical
  API ไม่ทำซ้ำถ้ามีข้อมูลครบแล้ว)
- มี smoke_test เหมือน Stage 1 (ทดสอบเรียกโมเดลที่เพิ่ง deploy จริง)

*** ข้อกำหนดก่อนรันไฟล์นี้ ***
- ต้องเพิ่ม scikit-learn, joblib, pandas ใน _PIP_ADDITIONAL_REQUIREMENTS
  (pandas มีอยู่แล้วจาก workshop 1, ส่วน scikit-learn/joblib เพิ่มไปแล้วจาก
  Stage 1 — ถ้าทำ Stage 1 ผ่านมาแล้วไม่ต้องทำอะไรเพิ่มตรงนี้)
- ต้องมี Airflow Connection "postgres_target" อยู่แล้ว (ตัวเดียวกับ
  workshop 1 stage 5 — host: postgres_target, port: 5432)
- ต้อง mount โฟลเดอร์ ./models ไว้แล้ว (ตัวเดียวกับที่ Stage 1 ใช้)

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ weather_pipeline_dag -> Unpause
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. รอบแรกจะใช้เวลานานกว่าปกติเล็กน้อยที่ task bootstrap_historical_data
   เพราะต้องดึงข้อมูลย้อนหลัง 14 วันจาก Open-Meteo Historical API
6. ดู Graph view -> เห็น flow เต็ม: create_tables -> bootstrap_historical_data
   -> extract_today_weather -> prepare_training_data -> train_model ->
   evaluate_model -> get_previous_rmse -> decide_deploy (branch) ->
   deploy_model -> smoke_test -> log_result (หรือทาง skip_deploy)
7. เปิด Logs ของ evaluate_model ดู RMSE ที่ได้ และ decide_deploy ดูว่า
   เทียบกับรอบก่อนแล้วผ่านไหม (รอบแรกที่ยังไม่มีรอบก่อนหน้า จะ deploy เสมอ)
8. ลอง trigger DAG ซ้ำอีกครั้ง (วันเดียวกัน) -> รอบนี้ extract_today_weather
   จะ "อัปเดต" แถวของวันนี้แทนที่จะสร้างซ้ำ (idempotent) และ decide_deploy
   จะเทียบกับ RMSE ที่เพิ่ง log ไปเมื่อรอบก่อน
"""

import math
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop2",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

POSTGRES_CONN_ID = "postgres_target"   # connection เดียวกับ workshop 1 stage 5
MODEL_NAME = "weather_bangkok"

# จังหวัดเดียว (ตามที่ตกลงกันไว้ — ไม่ทำหลายจังหวัดเพื่อไม่ให้ซ้ำกับ
# workshop 1 stage 4 ที่สอน fan-out ไปแล้ว)
PROVINCE = "กรุงเทพฯ"
LAT, LON = 13.7563, 100.5018

MIN_HISTORY_DAYS = 14   # จำนวนวันย้อนหลังขั้นต่ำที่ต้องมีก่อนเทรนได้
LAG_WINDOW = 3          # ใช้ข้อมูล 3 วันก่อนหน้าเป็น feature
HOLDOUT_SIZE = 3        # กันไว้ท้ายสุด 3 วันสำหรับวัด RMSE (ไม่ใช้เทรน)

MODEL_DIR = "/opt/airflow/models/weather_models"
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "current_model.pkl")

BANGKOK_TZ = ZoneInfo("Asia/Bangkok")

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS weather_history (
    date DATE NOT NULL,
    province VARCHAR(50) NOT NULL,
    temperature FLOAT NOT NULL,
    inserted_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (date, province)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    rmse FLOAT NOT NULL,
    deployed BOOLEAN NOT NULL,
    run_at TIMESTAMP DEFAULT NOW()
);
"""


def bootstrap_historical_data(**kwargs):
    """
    เช็คว่ามีข้อมูลย้อนหลังพอหรือยัง (idempotent) ถ้าไม่พอ ดึงจาก
    Open-Meteo Historical API มาเติมให้ครบ MIN_HISTORY_DAYS วัน
    """
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    count = hook.get_first(
        "SELECT COUNT(*) FROM weather_history WHERE province = %s;",
        parameters=(PROVINCE,),
    )[0]
    print(f"มีข้อมูลย้อนหลังอยู่แล้ว {count} วัน (ต้องการอย่างน้อย {MIN_HISTORY_DAYS} วัน)")

    if count >= MIN_HISTORY_DAYS:
        print("ข้อมูลครบแล้ว ข้ามการ bootstrap")
        return

    today = datetime.now(BANGKOK_TZ).date()
    start_date = today - timedelta(days=MIN_HISTORY_DAYS)
    end_date = today - timedelta(days=1)  # ไม่รวมวันนี้ (extract_today_weather จะดึงเอง)

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date.isoformat()}&end_date={end_date.isoformat()}"
        "&daily=temperature_2m_mean&timezone=Asia%2FBangkok"
    )
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    daily = response.json()["daily"]

    inserted = 0
    for date_str, temp in zip(daily["time"], daily["temperature_2m_mean"]):
        if temp is None:
            continue  # บางวันอาจยังไม่มีข้อมูลสมบูรณ์ ข้ามไป
        hook.run(
            """
            INSERT INTO weather_history (date, province, temperature)
            VALUES (%s, %s, %s)
            ON CONFLICT (date, province) DO NOTHING;
            """,
            parameters=(date_str, PROVINCE, temp),
        )
        inserted += 1

    print(f"Bootstrap เสร็จ: ดึงข้อมูลย้อนหลังมาเพิ่ม {inserted} วัน")


def extract_today_weather(**kwargs):
    """ดึงอุณหภูมิปัจจุบัน แล้วบันทึก/อัปเดตแถวของวันนี้ใน weather_history"""
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LAT}&longitude={LON}&current_weather=true"
        "&timezone=Asia%2FBangkok"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    current = response.json()["current_weather"]

    today = datetime.now(BANGKOK_TZ).date()
    temperature = current["temperature"]

    # ON CONFLICT DO UPDATE ทำให้ trigger ซ้ำวันเดียวกันได้โดยไม่สร้างแถวซ้ำ
    hook.run(
        """
        INSERT INTO weather_history (date, province, temperature)
        VALUES (%s, %s, %s)
        ON CONFLICT (date, province)
        DO UPDATE SET temperature = EXCLUDED.temperature, inserted_at = NOW();
        """,
        parameters=(today, PROVINCE, temperature),
    )
    print(f"บันทึกอุณหภูมิวันนี้ ({today}): {temperature} °C")


def prepare_training_data(**kwargs):
    """
    ดึงข้อมูลย้อนหลังทั้งหมดมาสร้าง feature (lag 3 วัน + moving average)
    และเตรียม feature ชุดล่าสุดไว้สำหรับทำนาย "พรุ่งนี้" ด้วย
    """
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    rows = hook.get_records(
        "SELECT date, temperature FROM weather_history "
        "WHERE province = %s ORDER BY date ASC;",
        parameters=(PROVINCE,),
    )
    df = pd.DataFrame(rows, columns=["date", "temperature"])
    print(f"ดึงข้อมูลย้อนหลังมาทั้งหมด {len(df)} วัน")

    temps = df["temperature"].tolist()

    features, targets = [], []
    for i in range(LAG_WINDOW, len(temps)):
        lag1, lag2, lag3 = temps[i - 1], temps[i - 2], temps[i - 3]
        moving_avg = (lag1 + lag2 + lag3) / 3
        features.append([lag1, lag2, lag3, moving_avg])
        targets.append(temps[i])

    print(f"สร้าง feature ได้ {len(features)} แถว (จากข้อมูลดิบ {len(temps)} วัน)")

    # feature ชุดล่าสุด สำหรับทำนายวันพรุ่งนี้ (ใช้ 3 วันล่าสุดที่มีจริง)
    last3 = temps[-LAG_WINDOW:]
    latest_features = [last3[-1], last3[-2], last3[-3], sum(last3) / LAG_WINDOW]

    ti.xcom_push(key="features", value=features)
    ti.xcom_push(key="targets", value=targets)
    ti.xcom_push(key="latest_features", value=latest_features)
    ti.xcom_push(key="latest_date", value=str(df["date"].iloc[-1]))


def train_model(**kwargs):
    """เทรน RandomForestRegressor โดยกันข้อมูลท้ายสุด HOLDOUT_SIZE แถวไว้วัดผล"""
    import joblib
    from sklearn.ensemble import RandomForestRegressor

    ti = kwargs["ti"]
    features = ti.xcom_pull(task_ids="prepare_training_data", key="features")
    targets = ti.xcom_pull(task_ids="prepare_training_data", key="targets")

    X_train = features[:-HOLDOUT_SIZE]
    y_train = targets[:-HOLDOUT_SIZE]
    X_holdout = features[-HOLDOUT_SIZE:]
    y_holdout = targets[-HOLDOUT_SIZE:]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    run_id = kwargs["run_id"].replace(":", "-").replace("+", "-")
    candidate_path = os.path.join(MODEL_DIR, f"candidate_{run_id}.pkl")
    joblib.dump(model, candidate_path)

    ti.xcom_push(key="candidate_model_path", value=candidate_path)
    ti.xcom_push(key="X_holdout", value=X_holdout)
    ti.xcom_push(key="y_holdout", value=y_holdout)
    print(f"เทรนโมเดลเสร็จ (train {len(X_train)} แถว, กันไว้วัดผล {len(X_holdout)} แถว)")
    print(f"บันทึกไว้ที่: {candidate_path}")


def evaluate_model(**kwargs):
    """วัด RMSE บนชุด holdout ที่กันไว้ (ไม่เคยใช้เทรน)"""
    import joblib

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")
    X_holdout = ti.xcom_pull(task_ids="train_model", key="X_holdout")
    y_holdout = ti.xcom_pull(task_ids="train_model", key="y_holdout")

    model = joblib.load(candidate_path)
    predictions = model.predict(X_holdout)

    squared_errors = [(p - a) ** 2 for p, a in zip(predictions, y_holdout)]
    rmse = math.sqrt(sum(squared_errors) / len(squared_errors))

    ti.xcom_push(key="rmse", value=rmse)
    print(f"ประเมินผลโมเดล: RMSE = {rmse:.4f} (ยิ่งต่ำยิ่งดี)")


def get_previous_rmse(**kwargs):
    """ดึง RMSE ของโมเดลที่ deploy ล่าสุดจาก Postgres มาเทียบ (champion)"""
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    row = hook.get_first(
        "SELECT rmse FROM model_metrics "
        "WHERE model_name = %s AND deployed = TRUE "
        "ORDER BY run_at DESC LIMIT 1;",
        parameters=(MODEL_NAME,),
    )
    previous_rmse = row[0] if row else None

    ti.xcom_push(key="previous_rmse", value=previous_rmse)
    if previous_rmse is None:
        print("ยังไม่เคยมีโมเดลที่ deploy มาก่อน (รอบแรก) — ถือว่ายังไม่มี champion ให้เทียบ")
    else:
        print(f"RMSE ของโมเดลรอบก่อนหน้า (champion ปัจจุบัน): {previous_rmse:.4f}")


def decide_deploy(**kwargs):
    """BranchPythonOperator: deploy เฉพาะเมื่อ RMSE ใหม่ดีกว่ารอบก่อน (หรือยังไม่มีรอบก่อน)"""
    ti = kwargs["ti"]
    rmse = ti.xcom_pull(task_ids="evaluate_model", key="rmse")
    previous_rmse = ti.xcom_pull(task_ids="get_previous_rmse", key="previous_rmse")

    if previous_rmse is None or rmse < previous_rmse:
        print(f"RMSE ใหม่ {rmse:.4f} ดีกว่า (หรือไม่มี) champion เดิม -> deploy_model")
        return "deploy_model"
    else:
        print(f"RMSE ใหม่ {rmse:.4f} ไม่ดีกว่า champion เดิม {previous_rmse:.4f} -> skip_deploy")
        return "skip_deploy"


def deploy_model(**kwargs):
    """คัดลอกโมเดลที่ชนะไปทับ current_model.pkl (จำลอง production)"""
    import shutil

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")

    shutil.copyfile(candidate_path, CURRENT_MODEL_PATH)
    print(f"Deploy สำเร็จ: {candidate_path} -> {CURRENT_MODEL_PATH}")

    return "deployed"


def skip_deploy(**kwargs):
    """ไม่ deploy เพราะยังสู้โมเดลเดิม (champion) ไม่ได้"""
    ti = kwargs["ti"]
    rmse = ti.xcom_pull(task_ids="evaluate_model", key="rmse")
    previous_rmse = ti.xcom_pull(task_ids="get_previous_rmse", key="previous_rmse")
    print(f"ข้าม deploy: RMSE ใหม่ {rmse:.4f} vs champion เดิม {previous_rmse:.4f}")
    print("โมเดลเดิม (current_model.pkl) ยังคงใช้งานต่อไป")

    return "skipped"


def smoke_test(**kwargs):
    """ทดสอบเรียกใช้งานโมเดลที่เพิ่ง deploy จริง ด้วย feature ของวันล่าสุด"""
    import joblib

    ti = kwargs["ti"]
    latest_features = ti.xcom_pull(task_ids="prepare_training_data", key="latest_features")
    latest_date = ti.xcom_pull(task_ids="prepare_training_data", key="latest_date")

    model = joblib.load(CURRENT_MODEL_PATH)
    forecast = model.predict([latest_features])[0]

    print("===== Smoke Test: เรียกใช้งานโมเดลที่เพิ่ง deploy =====")
    print(f"ข้อมูลล่าสุดที่มี: {latest_date} (feature: {latest_features})")
    print(f"พยากรณ์อุณหภูมิวันถัดไป: {forecast:.2f} °C")
    print("โมเดลใช้งานได้จริง พร้อมให้บริการ")


def log_result(**kwargs):
    """บันทึกผล RMSE รอบนี้ลง Postgres (ให้รอบถัดไปดึงไปเทียบเป็น champion)"""
    ti = kwargs["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    rmse = ti.xcom_pull(task_ids="evaluate_model", key="rmse")
    skip_result = ti.xcom_pull(task_ids="skip_deploy")
    deployed = skip_result is None  # ถ้า skip_deploy ไม่ได้รัน แปลว่า deploy ไปแล้ว

    hook.run(
        "INSERT INTO model_metrics (model_name, rmse, deployed) VALUES (%s, %s, %s);",
        parameters=(MODEL_NAME, rmse, deployed),
    )

    print("===== สรุปผล Stage 2 (weather pipeline) =====")
    print(f"RMSE: {rmse:.4f}")
    print(f"Deploy รอบนี้: {'ใช่' if deployed else 'ไม่ใช่'}")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="weather_pipeline_dag",
    default_args=default_args,
    description="Workshop 2 Stage 2: ทำนายอุณหภูมิวันถัดไปด้วย champion-challenger บน Postgres",
    schedule=None,   # ในงานจริงอาจตั้งเป็น "@daily" — ที่นี่ให้ trigger เองเพื่อทดสอบใน workshop
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["workshop2", "stage-2", "ml-pipeline"],
) as dag:

    create_tables_task = PostgresOperator(
        task_id="create_tables",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=CREATE_TABLES_SQL,
    )

    bootstrap_task = PythonOperator(
        task_id="bootstrap_historical_data",
        python_callable=bootstrap_historical_data,
    )

    extract_task = PythonOperator(
        task_id="extract_today_weather",
        python_callable=extract_today_weather,
    )

    prepare_task = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_training_data,
    )

    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    previous_rmse_task = PythonOperator(
        task_id="get_previous_rmse",
        python_callable=get_previous_rmse,
    )

    decide_task = BranchPythonOperator(task_id="decide_deploy", python_callable=decide_deploy)

    deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    skip_task = PythonOperator(task_id="skip_deploy", python_callable=skip_deploy)

    smoke_test_task = PythonOperator(task_id="smoke_test", python_callable=smoke_test)

    log_task = PythonOperator(
        task_id="log_result",
        python_callable=log_result,
        trigger_rule="none_failed_min_one_success",
    )

    # ลำดับการรันทั้งหมด (เส้นตรงเป็นหลัก ต่างจาก workshop 1 ที่เน้น parallel
    # เพราะไฟล์นี้โฟกัสที่แนวคิด champion-challenger ไม่ใช่เรื่อง fan-out)
    (
        create_tables_task
        >> bootstrap_task
        >> extract_task
        >> prepare_task
        >> train_task
        >> evaluate_task
        >> previous_rmse_task
        >> decide_task
    )
    decide_task >> deploy_task >> smoke_test_task >> log_task
    decide_task >> skip_task >> log_task