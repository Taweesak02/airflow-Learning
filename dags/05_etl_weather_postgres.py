"""
05_etl_weather_postgres.py
================
Workshop: เขียน DAG ด้วย Apache Airflow (ขั้นที่ 5 - ETL เต็มรูปแบบ)
*** ไฟล์หลักของ workshop นี้ ***

เป้าหมายของไฟล์นี้:
- รวมทุกแนวคิดจากไฟล์ 01-04 เข้าด้วยกันเป็น pipeline เดียวที่ใช้งานได้จริง
  (parallel extract, XCom, PythonOperator, และตอนนี้เพิ่ม PostgresOperator)
- ทำ ETL ครบ 3 ขั้น: Extract (ดึงจาก Open-Meteo หลายจังหวัดพร้อมกัน)
  -> Transform (แปลง/จัดรูปข้อมูลให้พร้อมเข้าฐานข้อมูล)
  -> Load (เขียนลง Postgres จริง)
- รู้จัก PostgresOperator (รัน SQL ตรงๆ ไม่ต้องเขียน Python เชื่อมเอง)
  และ PostgresHook (ใช้เชื่อมต่อ + insert ข้อมูลจาก PythonOperator)

*** ข้อกำหนดก่อนรันไฟล์นี้ (สำคัญมาก) ***
ต้องตั้งค่า Airflow Connection ชื่อ "postgres_target" ก่อน ผ่านเมนู
Admin -> Connections -> เพิ่มใหม่ (+) โดยใส่ค่าตามไฟล์ docker-compose.yaml:
    Connection Id   : postgres_target
    Connection Type : Postgres
    Host            : postgres_target
    Schema          : (ชื่อฐานข้อมูลที่ตั้งไว้ใน POSTGRES_DB ของ service postgres_target)
    Login           : (ค่า POSTGRES_USER ของ service postgres_target)
    Password        : (ค่า POSTGRES_PASSWORD ของ service postgres_target)
    Port            : 5432
ถ้ายังไม่ตั้งค่านี้ DAG จะ fail ทันทีที่ task ที่เชื่อมต่อ Postgres เริ่มรัน
(ดูรายละเอียดเพิ่มเติมได้ในคู่มือติดตั้ง หัวข้อ Checklist ก่อนเริ่มสอน)

วิธีทดสอบ:
1. ตั้งค่า Connection "postgres_target" ตามด้านบนให้เรียบร้อยก่อน
2. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
3. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
4. เปิด http://localhost:8080 -> หา DAG ชื่อ etl_weather_postgres_dag -> กด Unpause
5. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
6. ดู Graph view -> เห็นภาพรวม pipeline ทั้งหมด: extract 4 จังหวัดพร้อมกัน,
   สร้างตารางขนานไปด้วย, มารวมกันที่ load แล้วปิดท้ายด้วย verify_load
7. เปิด Logs ของ verify_load ดูว่าข้อมูลถูกเขียนลง Postgres จริงกี่แถว
8. (ทางเลือก) เชื่อมต่อ Postgres จากเครื่อง host ด้วย DBeaver/pgAdmin/psql ที่
   host=localhost, port=5433 เพื่อดูข้อมูลในตาราง weather_records ด้วยตาตัวเอง

หมายเหตุเรื่อง timezone:
- คอลัมน์ recorded_at ขอเวลาแบบไทย (Asia/Bangkok) ตรงจาก Open-Meteo ผ่าน
  parameter &timezone=Asia%2FBangkok ในโค้ดด้านล่าง
- คอลัมน์ ingested_at ยังเป็น UTC ตามค่า default ของ Postgres (NOW()) เพราะ
  container นี้ไม่ได้ตั้งค่า timezone ไว้ ดังนั้น 2 คอลัมน์นี้จะมี "เขตเวลา"
  ต่างกันโดยตั้งใจ ให้ระวังเวลาเทียบกันตรงๆ (ingested_at จะช้ากว่าเวลาไทยจริง
  7 ชั่วโมง) ถ้าต้องการดูเป็นเวลาไทยทั้งคู่ ใช้ query:
      SELECT *, ingested_at AT TIME ZONE 'Asia/Bangkok' AS ingested_at_th
      FROM weather_records;
- Airflow UI (Grid/Graph/Logs) เองก็แสดงเวลาเป็น UTC เป็นค่า default เช่นกัน
  เปลี่ยนได้ที่ไอคอนนาฬิกามุมขวาบนของหน้าเว็บ -> เลือก Asia/Bangkok
"""

from datetime import datetime, timedelta

import requests
from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.providers.postgres.operators.postgres import PostgresOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

POSTGRES_CONN_ID = "postgres_target"
TABLE_NAME = "weather_records"

PROVINCES = [
    {"name": "กรุงเทพฯ", "slug": "bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "เชียงใหม่", "slug": "chiang_mai", "lat": 18.7883, "lon": 98.9853},
    {"name": "ขอนแก่น", "slug": "khon_kaen", "lat": 16.4419, "lon": 102.8360},
    {"name": "ภูเก็ต", "slug": "phuket", "lat": 7.8804, "lon": 98.3923},
]

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id SERIAL PRIMARY KEY,
    province VARCHAR(50) NOT NULL,
    temperature FLOAT NOT NULL,
    windspeed FLOAT,
    weathercode INT,
    temperature_category VARCHAR(20),
    recorded_at TIMESTAMP,
    ingested_at TIMESTAMP DEFAULT NOW()
);
"""


# -----------------------------------------------------------------
# Extract: ดึงข้อมูลอากาศดิบจาก Open-Meteo ทีละจังหวัด (รันพร้อมกัน)
# -----------------------------------------------------------------
def extract_weather(name, slug, lat, lon, **kwargs):
    """ดึงข้อมูลอากาศปัจจุบันของจังหวัดหนึ่ง แล้ว push เป็น dict เข้า XCom"""
    ti = kwargs["ti"]
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}&current_weather=true"
        "&timezone=Asia%2FBangkok"
    )
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    current = response.json()["current_weather"]

    weather_data = {
        "province": name,
        "temperature": current["temperature"],
        "windspeed": current["windspeed"],
        "weathercode": current["weathercode"],
        "recorded_at": current["time"],
    }
    print(f"ดึงข้อมูล {name} สำเร็จ: {weather_data}")

    ti.xcom_push(key="weather_data", value=weather_data)


# -----------------------------------------------------------------
# Transform: รวมข้อมูลทุกจังหวัด + เพิ่ม field ที่คำนวณใหม่ (temperature_category)
# -----------------------------------------------------------------
def transform_weather(**kwargs):
    """
    ดึงข้อมูลดิบจากทุก extract task มารวมเป็น list เดียว
    พร้อมเพิ่ม field 'temperature_category' ที่ไม่ได้มาจาก API ตรงๆ
    (ตัวอย่างการ transform ง่ายๆ ที่พบบ่อยในงาน ETL จริง)
    """
    ti = kwargs["ti"]
    records = []

    for province in PROVINCES:
        task_id = f"extract_{province['slug']}"
        data = ti.xcom_pull(task_ids=task_id, key="weather_data")

        temperature = data["temperature"]
        if temperature >= 35:
            category = "ร้อนจัด"
        elif temperature >= 30:
            category = "ร้อน"
        elif temperature >= 25:
            category = "อากาศดี"
        else:
            category = "เย็น"

        data["temperature_category"] = category
        records.append(data)
        print(f"แปลงข้อมูล {data['province']}: {temperature} °C -> {category}")

    ti.xcom_push(key="transformed_records", value=records)
    print(f"แปลงข้อมูลครบ {len(records)} จังหวัด")


# -----------------------------------------------------------------
# Load: เขียนข้อมูลที่แปลงแล้วลง Postgres จริง
# -----------------------------------------------------------------
def load_to_postgres(**kwargs):
    """ใช้ PostgresHook เชื่อมต่อฐานข้อมูลปลายทาง แล้ว insert ทีละแถว"""
    ti = kwargs["ti"]
    records = ti.xcom_pull(task_ids="transform_weather", key="transformed_records")

    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    rows = [
        (
            r["province"],
            r["temperature"],
            r["windspeed"],
            r["weathercode"],
            r["temperature_category"],
            r["recorded_at"],
        )
        for r in records
    ]

    hook.insert_rows(
        table=TABLE_NAME,
        rows=rows,
        target_fields=[
            "province",
            "temperature",
            "windspeed",
            "weathercode",
            "temperature_category",
            "recorded_at",
        ],
    )
    print(f"บันทึกลง Postgres สำเร็จ {len(rows)} แถว")


# -----------------------------------------------------------------
# Verify: ยืนยันผลลัพธ์ (สรุปให้ดูว่าข้อมูลเข้าฐานข้อมูลจริง)
# -----------------------------------------------------------------
def verify_load(**kwargs):
    """
    Query กลับจาก Postgres มาแสดงผล เป็นเหมือน 'smoke test' ปิดท้าย pipeline
    ยืนยันว่าข้อมูลที่ insert ไปจริงๆ อ่านกลับมาได้ถูกต้อง
    """
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    total_count = hook.get_first(f"SELECT COUNT(*) FROM {TABLE_NAME};")[0]
    print(f"จำนวนแถวทั้งหมดในตาราง {TABLE_NAME}: {total_count}")

    latest_rows = hook.get_records(
        f"""
        SELECT province, temperature, temperature_category, ingested_at
        FROM {TABLE_NAME}
        ORDER BY id DESC
        LIMIT 4;
        """
    )
    print("===== แถวล่าสุดที่เพิ่ง insert =====")
    for row in latest_rows:
        province, temperature, category, ingested_at = row
        print(f"{province}: {temperature} °C ({category}) - บันทึกเมื่อ {ingested_at}")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="etl_weather_postgres_dag",
    default_args=default_args,
    description="ขั้นที่ 5: ETL เต็มรูปแบบ - Extract จาก Open-Meteo -> Transform -> Load เข้า Postgres",
    schedule=None,                  # None = ไม่ตั้งเวลาอัตโนมัติ ต้อง trigger เอง (เหมาะกับตอนทดสอบ)
    start_date=datetime(2026, 8, 1),
    catchup=False,                  # ไม่ต้องรันย้อนหลังตั้งแต่ start_date
    tags=["workshop", "step-05"],
) as dag:

    start = EmptyOperator(task_id="start")

    # สร้างตารางถ้ายังไม่มี ใช้ PostgresOperator รัน SQL ตรงๆ (ไม่ต้องเขียน
    # Python เชื่อมต่อเอง) รันคู่ขนานไปกับการ extract ได้เลยเพราะไม่เกี่ยวข้องกัน
    create_table_task = PostgresOperator(
        task_id="create_table",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=CREATE_TABLE_SQL,
    )

    # Extract แบบ fan-out: ดึงข้อมูล 4 จังหวัดพร้อมกัน (เหมือนไฟล์ 04)
    extract_tasks = []
    for province in PROVINCES:
        extract_task = PythonOperator(
            task_id=f"extract_{province['slug']}",
            python_callable=extract_weather,
            op_kwargs={
                "name": province["name"],
                "slug": province["slug"],
                "lat": province["lat"],
                "lon": province["lon"],
            },
        )
        extract_tasks.append(extract_task)

    # จุดรวมหลัง extract ครบทุกจังหวัด (fan-in)
    join_extract = EmptyOperator(task_id="join_extract")

    transform_task = PythonOperator(
        task_id="transform_weather",
        python_callable=transform_weather,
    )

    load_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres,
    )

    verify_task = PythonOperator(
        task_id="verify_load",
        python_callable=verify_load,
    )

    # ลำดับการรันทั้งหมด:
    # start -> extract 4 จังหวัดพร้อมกัน -> join_extract -> transform -> load
    # create_table_task รันคู่ขนานไปกับฝั่ง extract แต่ load ต้องรอทั้งสองฝั่งเสร็จ
    start >> extract_tasks >> join_extract >> transform_task
    start >> create_table_task
    [transform_task, create_table_task] >> load_task >> verify_task