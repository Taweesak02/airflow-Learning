"""
03_branching_dag.py
================
Workshop: เขียน DAG ด้วย Apache Airflow (ขั้นที่ 3 - Branching)

เป้าหมายของไฟล์นี้:
- เข้าใจว่า BranchPythonOperator ใช้เลือกเส้นทางการรันแบบมีเงื่อนไขได้อย่างไร
- เห็นว่า task ที่ "ไม่ถูกเลือก" จะขึ้นสถานะ skipped (สีเทา/ส้มอ่อน) ไม่ใช่ failed
- เข้าใจ trigger_rule ที่ต้องใช้เป็นพิเศษเมื่อ task ปลายทางรอรับจากหลาย branch
  ที่ไม่ได้รันครบทุกเส้นทาง

แนวคิดสำคัญ:
BranchPythonOperator ทำงานคล้าย PythonOperator ทุกอย่าง ต่างกันตรงที่ function
ที่ใช้ต้อง "return ชื่อ task_id" ของ task ถัดไปที่ต้องการให้รัน (เป็น string
หรือ list ของ string ก็ได้) Airflow จะรันเฉพาะ task ที่ถูก return กลับมา
ส่วน task อื่นที่อยู่ระดับเดียวกันแต่ไม่ถูกเลือก จะถูกข้ามไปเป็นสถานะ skipped

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ branching_dag -> กด Unpause (toggle)
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. ดู Graph view -> สังเกตว่ามีแค่ 1 ใน 2 ทาง (high_branch_task หรือ
   low_branch_task) ที่ขึ้นสีเขียว (success) ส่วนอีกทางจะขึ้นสีเทา/ส้มอ่อน
   (skipped) เพราะไม่ถูกเลือกให้รัน
6. ลอง Trigger ซ้ำหลายๆ ครั้ง จะเห็นว่าบางครั้งไปทาง high บางครั้งไปทาง low
   สลับกันไป เพราะในตัวอย่างนี้ใช้ตัวเลขสุ่มเป็นตัวตัดสินใจ
7. เปิด Logs ของ join_task ดูว่าสรุปผลถูกต้องตรงกับทางที่ถูกเลือกจริง
"""

import random
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def check_condition(**kwargs):
    """
    Task ตัดสินใจ (BranchPythonOperator ต้องใช้ function แบบนี้เท่านั้น)
    สุ่มตัวเลข 1-100 มาเป็นตัวอย่างเงื่อนไข (ในงานจริงอาจเป็นการเช็คว่า
    วันนี้เป็นวันธรรมดา/วันหยุด, เช็คว่าไฟล์มาถึงหรือยัง, เช็คค่าจาก API ฯลฯ)

    สิ่งที่ return ต้องเป็น "ชื่อ task_id" ของ task ถัดไปที่ต้องการให้รัน
    ห้าม return ค่าอื่นที่ไม่ใช่ task_id ที่มีอยู่จริงใน DAG นี้ ไม่งั้นจะ error
    """
    ti = kwargs["ti"]
    number = random.randint(1, 100)
    print(f"สุ่มตัวเลขได้: {number}")

    ti.xcom_push(key="random_number", value=number)

    if number >= 50:
        print(f"{number} >= 50 -> เลือกเส้นทาง high_branch_task")
        return "high_branch_task"
    else:
        print(f"{number} < 50 -> เลือกเส้นทาง low_branch_task")
        return "low_branch_task"


def high_branch(**kwargs):
    """Task ฝั่ง 'high' จะถูกรันก็ต่อเมื่อ check_condition เลือกทางนี้เท่านั้น"""
    ti = kwargs["ti"]
    number = ti.xcom_pull(task_ids="check_condition", key="random_number")
    print(f"เข้าสู่เส้นทาง HIGH: ตัวเลขที่สุ่มได้คือ {number} (>= 50)")
    print("จำลองการประมวลผลสำหรับกรณีค่าสูง...")
    return "high"


def low_branch(**kwargs):
    """Task ฝั่ง 'low' จะถูกรันก็ต่อเมื่อ check_condition เลือกทางนี้เท่านั้น"""
    ti = kwargs["ti"]
    number = ti.xcom_pull(task_ids="check_condition", key="random_number")
    print(f"เข้าสู่เส้นทาง LOW: ตัวเลขที่สุ่มได้คือ {number} (< 50)")
    print("จำลองการประมวลผลสำหรับกรณีค่าต่ำ...")
    return "low"


def join_task(**kwargs):
    """
    Task สุดท้ายที่รวมทั้งสองเส้นทางกลับมาเป็นเส้นเดียว
    ใช้ xcom_pull() แบบ task_ids เป็น list เพื่อดึงค่าจากทั้งสองทาง
    โดยทางที่ไม่ถูกรัน (skipped) จะ pull ค่าออกมาเป็น None โดยอัตโนมัติ
    """
    ti = kwargs["ti"]
    number = ti.xcom_pull(task_ids="check_condition", key="random_number")
    high_result = ti.xcom_pull(task_ids="high_branch_task")
    low_result = ti.xcom_pull(task_ids="low_branch_task")

    print("===== สรุปผล (Join) =====")
    print(f"ตัวเลขที่สุ่มได้ตอนแรก: {number}")
    print(f"ผลจาก high_branch_task: {high_result}")
    print(f"ผลจาก low_branch_task: {low_result}")

    chosen = "high" if high_result == "high" else "low"
    print(f"สรุป: DAG run นี้เลือกเดินเส้นทาง -> {chosen}")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="branching_dag",
    default_args=default_args,
    description="ขั้นที่ 3: เลือกเส้นทางการรันแบบมีเงื่อนไขด้วย BranchPythonOperator",
    schedule=None,                  # None = ไม่ตั้งเวลาอัตโนมัติ ต้อง trigger เอง (เหมาะกับตอนทดสอบ)
    start_date=datetime(2026, 8, 1),
    catchup=False,                  # ไม่ต้องรันย้อนหลังตั้งแต่ start_date
    tags=["workshop", "step-03"],
) as dag:

    check_condition_task = BranchPythonOperator(
        task_id="check_condition",
        python_callable=check_condition,
    )

    high_branch_task = PythonOperator(
        task_id="high_branch_task",
        python_callable=high_branch,
    )

    low_branch_task = PythonOperator(
        task_id="low_branch_task",
        python_callable=low_branch,
    )

    # trigger_rule="none_failed_min_one_success" สำคัญมาก:
    # ค่า default ของ trigger_rule คือ "all_success" ซึ่งต้องการให้ task
    # ก่อนหน้า "ทุกตัว" สำเร็จก่อนถึงจะรัน แต่ในกรณีนี้มีตัวหนึ่งที่ถูก skip
    # เสมอ (ไม่ใช่ success) จึงต้องเปลี่ยน trigger_rule ให้ join_task รันได้
    # ตราบใดที่อย่างน้อย 1 ทางสำเร็จ และไม่มีทางไหน failed จริงๆ
    join = PythonOperator(
        task_id="join_task",
        python_callable=join_task,
        trigger_rule="none_failed_min_one_success",
    )

    # กำหนดลำดับการรัน: check_condition เลือกว่าจะไปทางไหน
    # ทั้งสองทางวิ่งกลับมารวมกันที่ join_task
    check_condition_task >> [high_branch_task, low_branch_task] >> join