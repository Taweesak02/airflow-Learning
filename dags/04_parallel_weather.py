"""
04_parallel_weather.py
================
Workshop: เขียน DAG ด้วย Apache Airflow (ขั้นที่ 4 - Parallel Tasks)

เป้าหมายของไฟล์นี้:
- เข้าใจว่า task ที่ไม่มี dependency ระหว่างกันจะถูก Airflow รันพร้อมกัน (parallel)
  โดยอัตโนมัติ ไม่ต้องสั่งอะไรพิเศษ แค่ไม่ผูก >> ระหว่างกันเท่านั้น
- ฝึกรูปแบบ fan-out (task เดียวแตกออกหลายทาง) และ fan-in (หลายทางกลับมารวม
  เป็นทางเดียว) ซึ่งเป็นรูปแบบที่พบบ่อยมากในงาน ETL จริง
- รู้จัก EmptyOperator ใช้เป็นจุดรวม (join point) ที่ไม่ต้องทำอะไรจริง
- เริ่มดึงข้อมูลจาก API จริง (Open-Meteo) แทนข้อมูลจำลอง เตรียมพร้อมสำหรับ
  ไฟล์ 05 ที่จะทำ ETL เข้า Postgres

แนวคิดสำคัญ:
Open-Meteo (https://open-meteo.com) เป็น API พยากรณ์อากาศที่ไม่ต้องใช้ API key
เรียกได้ฟรีไม่จำกัด เหมาะกับ workshop ที่มีคนใช้งานพร้อมกันหลายเครื่อง

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ parallel_weather_dag -> กด Unpause
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. ดู Graph view -> สังเกตว่า start แตกออกเป็น 4 เส้นพร้อมกัน (fetch_bangkok,
   fetch_chiang_mai, fetch_khon_kaen, fetch_phuket) แล้วค่อยมารวมที่ join
   ก่อนไป summarize_weather
6. ดู Gantt view -> จะเห็นแท่งของทั้ง 4 task ทับซ้อนช่วงเวลากัน (รันพร้อมกันจริง)
   ต่างจากไฟล์ 01-03 ที่ task รันเรียงต่อกันทีละตัว
7. เปิด Logs ของ summarize_weather ดูสรุปว่าจังหวัดไหนร้อนสุด/เย็นสุดตอนนี้
"""

from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# รายชื่อจังหวัดและพิกัดที่จะดึงข้อมูลอากาศ (เพิ่ม/ลดได้ตามต้องการ)
PROVINCES = [
    {"name": "กรุงเทพฯ", "slug": "bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "เชียงใหม่", "slug": "chiang_mai", "lat": 18.7883, "lon": 98.9853},
    {"name": "ขอนแก่น", "slug": "khon_kaen", "lat": 16.4419, "lon": 102.8360},
    {"name": "ภูเก็ต", "slug": "phuket", "lat": 7.8804, "lon": 98.3923},
]


def fetch_weather(name, lat, lon, **kwargs):
    """
    ดึงอุณหภูมิปัจจุบันของจังหวัดหนึ่งจาก Open-Meteo API
    task แบบนี้จะถูกสร้างซ้ำ 1 ชุดต่อ 1 จังหวัด (ดูส่วนสร้าง DAG ด้านล่าง)
    ทุก task ไม่ได้ผูก dependency ระหว่างกันเอง จึงรันพร้อมกันได้
    """
    ti = kwargs["ti"]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()

    temperature = data["current_weather"]["temperature"]
    print(f"{name}: อุณหภูมิปัจจุบัน {temperature} °C")

    ti.xcom_push(key="temperature", value=temperature)


def summarize_weather(**kwargs):
    """
    รวบรวมผลจากทุก fetch task มาสรุป: จังหวัดไหนร้อนสุด/เย็นสุด
    ใช้ task_ids ระบุชื่อ task ต้นทางแต่ละตัวตอน pull ค่าออกมาทีละจังหวัด
    """
    ti = kwargs["ti"]
    results = {}

    for province in PROVINCES:
        task_id = f"fetch_{province['slug']}"
        temperature = ti.xcom_pull(task_ids=task_id, key="temperature")
        results[province["name"]] = temperature
        print(f"{province['name']}: {temperature} °C")

    hottest = max(results, key=results.get)
    coolest = min(results, key=results.get)

    print("===== สรุปผล =====")
    print(f"จังหวัดที่ร้อนที่สุดตอนนี้: {hottest} ({results[hottest]} °C)")
    print(f"จังหวัดที่เย็นที่สุดตอนนี้: {coolest} ({results[coolest]} °C)")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="parallel_weather_dag",
    default_args=default_args,
    description="ขั้นที่ 4: ดึงอากาศหลายจังหวัดพร้อมกัน (fan-out) แล้วสรุปผล (fan-in)",
    schedule=None,                  # None = ไม่ตั้งเวลาอัตโนมัติ ต้อง trigger เอง (เหมาะกับตอนทดสอบ)
    start_date=datetime(2026, 8, 1),
    catchup=False,                  # ไม่ต้องรันย้อนหลังตั้งแต่ start_date
    tags=["workshop", "step-04"],
) as dag:

    start = EmptyOperator(task_id="start")

    # สร้าง task ดึงอากาศ 1 ชุดต่อ 1 จังหวัด โดยไม่ผูก dependency ระหว่างกันเอง
    # ทำให้ Airflow รันทั้งหมดนี้พร้อมกัน (parallel) โดยอัตโนมัติ
    fetch_tasks = []
    for province in PROVINCES:
        fetch_task = PythonOperator(
            task_id=f"fetch_{province['slug']}",
            python_callable=fetch_weather,
            op_kwargs={
                "name": province["name"],
                "lat": province["lat"],
                "lon": province["lon"],
            },
        )
        fetch_tasks.append(fetch_task)

    # join เป็นจุดรวมที่ไม่ทำอะไรจริง แค่รอให้ทุก fetch task เสร็จก่อน
    # ค่อยปล่อยให้ summarize_weather รันต่อ
    join = EmptyOperator(task_id="join")

    summarize_task = PythonOperator(
        task_id="summarize_weather",
        python_callable=summarize_weather,
    )

    # กำหนดลำดับการรัน: start -> แตกออกหลายทางพร้อมกัน -> รวมที่ join -> summarize
    start >> fetch_tasks >> join >> summarize_task