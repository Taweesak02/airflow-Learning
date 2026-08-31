"""
02_xcom_dag.py
================
Workshop: เขียน DAG ด้วย Apache Airflow (ขั้นที่ 2 - XCom)

เป้าหมายของไฟล์นี้:
- เข้าใจว่า XCom (Cross-Communication) คืออะไร และใช้ส่งข้อมูลระหว่าง task อย่างไร
- ฝึกใช้ ti.xcom_push() และ ti.xcom_pull() แบบระบุชื่อ (key) เอง
- เห็นว่า return value ของ PythonOperator ก็ถูกเก็บเข้า XCom โดยอัตโนมัติ (มาจาก step 01)
- จำลอง pattern แบบ mini-ETL: extract -> transform -> load

แนวคิดสำคัญ:
XCom ("Cross-Communication") คือกลไกของ Airflow ที่ให้ task ส่งข้อมูล "ชิ้นเล็กๆ"
ถึงกันได้ (ไม่เหมาะกับไฟล์ใหญ่หรือ DataFrame ทั้งก้อน) โดยข้อมูลจะถูกเก็บลง
Airflow metadata database แล้ว task ถัดไปดึงออกมาใช้ผ่าน task instance (ti)

วิธีทดสอบ:
1. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
2. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
3. เปิด http://localhost:8080 -> หา DAG ชื่อ xcom_dag -> กด Unpause (toggle)
4. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
5. คลิกเข้าไปดู Graph view -> คลิกแต่ละ task -> ดู Logs
6. ลองดูเมนู "XCom" ที่แถบด้านบนของหน้า DAG run (หรือคลิก task -> XCom)
   จะเห็นค่าที่แต่ละ task push เข้าไปจริงๆ พร้อม key ที่ตั้งชื่อไว้
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def extract(**kwargs):
    """
    Task 1 (จำลอง Extract): "ดึง" ข้อมูลตัวเลขมาสามตัว
    ในงานจริงตรงนี้อาจเป็นการเรียก API หรืออ่านไฟล์
    ใช้ ti.xcom_push() เพื่อส่งข้อมูลออกไปแบบระบุชื่อ (key) เอง
    """
    ti = kwargs["ti"]
    raw_numbers = [15, 42, 8]
    print(f"ดึงข้อมูลตัวเลขมาได้: {raw_numbers}")

    ti.xcom_push(key="raw_numbers", value=raw_numbers)
    print("push ค่า raw_numbers เข้า XCom เรียบร้อย")


def transform(**kwargs):
    """
    Task 2 (จำลอง Transform): ดึงค่าที่ extract ส่งมา แล้วแปลงข้อมูล
    ใช้ ti.xcom_pull() ระบุ task_ids ของ task ต้นทาง และ key ที่ตรงกัน
    """
    ti = kwargs["ti"]
    raw_numbers = ti.xcom_pull(task_ids="extract_task", key="raw_numbers")
    print(f"ดึงค่าจาก extract_task มาได้: {raw_numbers}")

    doubled = [n * 2 for n in raw_numbers]
    total = sum(doubled)
    print(f"แปลงข้อมูล (คูณ 2 ทีละตัว): {doubled}")
    print(f"รวมผลลัพธ์ทั้งหมด: {total}")

    ti.xcom_push(key="doubled_numbers", value=doubled)
    ti.xcom_push(key="total", value=total)
    print("push ค่า doubled_numbers และ total เข้า XCom เรียบร้อย")


def load(**kwargs):
    """
    Task 3 (จำลอง Load): ดึงผลลัพธ์สุดท้ายมาแสดง (ในงานจริงอาจเป็นการบันทึกลง DB)
    สังเกตว่า pull ได้ทั้งจาก extract_task โดยตรง และจาก transform_task
    (แปลว่า task ปลายทางสามารถดึงค่าจาก task ไหนก็ได้ ไม่จำเป็นต้องเป็น task ก่อนหน้าติดกัน)
    """
    ti = kwargs["ti"]
    raw_numbers = ti.xcom_pull(task_ids="extract_task", key="raw_numbers")
    doubled = ti.xcom_pull(task_ids="transform_task", key="doubled_numbers")
    total = ti.xcom_pull(task_ids="transform_task", key="total")

    print("===== สรุปผล (Load) =====")
    print(f"ข้อมูลดิบจาก extract_task : {raw_numbers}")
    print(f"ข้อมูลหลังแปลงจาก transform_task : {doubled}")
    print(f"ผลรวม : {total}")
    print("บันทึกผลลัพธ์เรียบร้อย (จำลอง)")

    return total  # ค่านี้จะถูกเก็บเข้า XCom อัตโนมัติด้วย key ชื่อ "return_value"


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="xcom_dag",
    default_args=default_args,
    description="ขั้นที่ 2: ส่งข้อมูลระหว่าง task ด้วย XCom (จำลอง extract -> transform -> load)",
    schedule=None,                  # None = ไม่ตั้งเวลาอัตโนมัติ ต้อง trigger เอง (เหมาะกับตอนทดสอบ)
    start_date=datetime(2026, 8, 1),
    catchup=False,                  # ไม่ต้องรันย้อนหลังตั้งแต่ start_date
    tags=["workshop", "step-02"],
) as dag:

    extract_task = PythonOperator(
        task_id="extract_task",
        python_callable=extract,
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=transform,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load,
    )

    # กำหนดลำดับการรัน: extract -> transform -> load เรียงกันตามลำดับ
    extract_task >> transform_task >> load_task