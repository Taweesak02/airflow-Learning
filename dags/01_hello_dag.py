"""
01_hello_dag.py
================
Workshop: เขียน DAG ด้วย Apache Airflow (ขั้นที่ 1 - Hello DAG)

เป้าหมายของไฟล์นี้:
- ทำความเข้าใจโครงสร้างพื้นฐานของ DAG
- ใช้ BashOperator และ PythonOperator
- กำหนดลำดับการรันของ task ด้วย >>

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ hello_dag -> กด Unpause (toggle)
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. คลิกเข้าไปดู Graph view -> คลิกแต่ละ task -> ดู Logs
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def say_hello(**kwargs):
    """
    ฟังก์ชัน Python ธรรมดาที่ Airflow จะเรียกผ่าน PythonOperator
    **kwargs คือ context ที่ Airflow ส่งเข้ามาให้อัตโนมัติ
    (เช่น ds = execution date, ti = task instance ที่ใช้ส่ง XCom)
    """
    execution_date = kwargs["ds"]
    print(f"สวัสดีจาก Airflow! วันที่รัน (execution date) คือ: {execution_date}")
    print("นี่คือ task ที่ 2 ซึ่งรันต่อจาก task แรก (print_date) สำเร็จแล้ว")
    return "hello-done"


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="hello_dag",
    default_args=default_args,
    description="DAG แรก: BashOperator + PythonOperator เรียงลำดับง่ายๆ",
    schedule=None,                  # None = ไม่ตั้งเวลาอัตโนมัติ ต้อง trigger เอง (เหมาะกับตอนทดสอบ)
    start_date=datetime(2026, 8, 1),
    catchup=False,                  # ไม่ต้องรันย้อนหลังตั้งแต่ start_date
    tags=["workshop", "step-01"],
) as dag:

    # Task 1: รันคำสั่ง shell ธรรมดา แสดงวันเวลาปัจจุบันของ container
    print_date = BashOperator(
        task_id="print_date",
        bash_command="date",
    )

    # Task 2: เรียกฟังก์ชัน Python ที่นิยามไว้ด้านบน
    hello_task = PythonOperator(
        task_id="hello_task",
        python_callable=say_hello,
    )

    # กำหนดลำดับการรัน: print_date ต้องรันเสร็จก่อน แล้วค่อยรัน hello_task
    print_date >> hello_task