"""
ml_01_iris_pipeline.py
================
Workshop 2: ML Pipeline บน Apache Airflow (Stage 1 - sklearn iris)

เป้าหมายของไฟล์นี้:
- เห็นโครงสร้างเต็มของ ML pipeline แบบไม่มีตัวแปรกวนใจ: ข้อมูลนิ่ง, ไม่พึ่ง
  network, ผลลัพธ์เหมือนเดิมทุกครั้งที่ trigger (deterministic)
- ครบ flow: Extract -> Prepare -> Train -> Evaluate -> Branch -> Deploy/Skip
  -> Smoke Test -> Log ซึ่งเป็นโครงเดียวกับที่จะใช้ใน Stage 2 (Open-Meteo)
  ที่ซับซ้อนกว่า
- ฝึกแนวคิด "deploy เฉพาะเมื่อผ่านเกณฑ์" (ที่นี่ใช้เกณฑ์คงที่ accuracy > 0.9
  ส่วน Stage 2 จะยกระดับเป็นเทียบกับโมเดลรอบก่อนหน้า)
- (Stage 3 เสริม) smoke_test เรียกใช้งานโมเดลที่เพิ่ง deploy จริงด้วยข้อมูล
  ตัวอย่าง ยืนยันว่าไฟล์โมเดลใช้งานได้จริง ไม่ใช่แค่เขียนไฟล์สำเร็จเฉยๆ

*** ข้อกำหนดก่อนรันไฟล์นี้ ***
ต้องเพิ่ม scikit-learn และ joblib ใน _PIP_ADDITIONAL_REQUIREMENTS ของ
docker-compose.yaml ก่อน (ไฟล์เดิมของ workshop 1 ยังไม่มี 2 ตัวนี้):
    _PIP_ADDITIONAL_REQUIREMENTS: >-
      beautifulsoup4 pandas sqlalchemy psycopg2-binary requests
      apache-airflow-providers-postgres scikit-learn joblib
และต้อง mount โฟลเดอร์ ./models (สำหรับเก็บไฟล์โมเดลแบบถาวร ใช้ร่วมกับ
เว็บทดสอบ model_api ด้วย):
    volumes:
      - ${AIRFLOW_PROJ_DIR:-.}/models:/opt/airflow/models
แล้วรัน docker compose up -d ใหม่ ไม่งั้น task จะ fail ตั้งแต่ import
(ดูไฟล์ docker-compose.yaml ที่อัปเดตแล้วประกอบ)

*** เว็บทดสอบโมเดล (ตัวอย่างขั้นสูง แยกต่างหาก) ***
มี service model_api (FastAPI) แยกจาก Airflow ที่เสิร์ฟโมเดลล่าสุดที่ DAG
นี้ deploy ไว้ ให้ทดสอบผ่านเว็บได้ที่ http://localhost:8001 หลังจาก
docker compose up -d และรัน DAG นี้จนถึง deploy_model สำเร็จอย่างน้อย 1
ครั้ง (รายละเอียดอยู่ที่โฟลเดอร์ ./model_service/)

วิธีทดสอบ:
1. เพิ่ม dependency ตามด้านบน แล้ว docker compose up -d ใหม่
2. copy ไฟล์นี้ไปวางในโฟลเดอร์ ./dags/
3. รอ Airflow scheduler สแกนเจอ (ไม่เกิน 30 วินาที)
4. เปิด http://localhost:8080 -> หา DAG ชื่อ iris_pipeline_dag -> กด Unpause
5. กด Trigger (ปุ่มสามเหลี่ยม ▶) เพื่อรันด้วยมือ
6. ดู Graph view -> เห็น flow เต็ม: extract -> prepare -> train -> evaluate
   -> decide_deploy (branch) -> deploy_model -> smoke_test -> log_result
   (หรือทาง skip_deploy -> log_result ถ้าไม่ผ่านเกณฑ์)
7. เปิด Logs ของ evaluate_model ดู accuracy ที่ได้ (ควรได้สูงมาก เพราะ iris
   เป็น dataset ง่าย มักได้ accuracy > 0.9 ทุกครั้ง แปลว่า deploy_model ควร
   ถูกเลือกเสมอ ไม่ใช่ skip_deploy)
8. เปิด Logs ของ log_result ดูสรุปผลรอบนี้
9. เปิด Logs ของ smoke_test (ถ้า branch ไปทาง deploy) ดูว่าโมเดลทายถูก
   กี่แถวจาก 3 แถวตัวอย่าง (iris ง่ายมาก ปกติควรถูกทั้ง 3)
"""

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

# -----------------------------------------------------------------
# ค่าเริ่มต้นที่ใช้ร่วมกันทุก task ใน DAG นี้
# -----------------------------------------------------------------
default_args = {
    "owner": "workshop2",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

# ที่เก็บไฟล์โมเดล — mount ไว้ที่ ./models บนเครื่อง host (ดู docker-compose.yaml)
# ใช้ path นี้แทน /tmp เพื่อให้ (1) ข้อมูลอยู่ถาวรข้าม container restart และ
# (2) service model_api (FastAPI) อ่านไฟล์เดียวกันนี้ได้ ผ่าน volume ร่วมกัน
MODEL_DIR = "/opt/airflow/models/iris_models"
CURRENT_MODEL_PATH = os.path.join(MODEL_DIR, "current_model.pkl")
ACCURACY_THRESHOLD = 0.9


def extract_data(**kwargs):
    """โหลดข้อมูล iris จาก sklearn ตรงๆ ไม่ต้องพึ่ง network หรือไฟล์ภายนอก"""
    from sklearn.datasets import load_iris

    ti = kwargs["ti"]
    iris = load_iris()

    # XCom เก็บได้แค่ข้อมูลที่ serialize เป็น JSON ได้ จึงแปลง numpy array
    # เป็น list ธรรมดาก่อน push (iris มีแค่ 150 แถว ขนาดเล็กมาก ไม่มีปัญหา)
    ti.xcom_push(key="features", value=iris.data.tolist())
    ti.xcom_push(key="targets", value=iris.target.tolist())
    print(f"โหลดข้อมูล iris สำเร็จ: {len(iris.data)} แถว, {iris.data.shape[1]} feature")


def prepare_data(**kwargs):
    """แบ่งข้อมูลเป็น train/test set"""
    from sklearn.model_selection import train_test_split

    ti = kwargs["ti"]
    features = ti.xcom_pull(task_ids="extract_data", key="features")
    targets = ti.xcom_pull(task_ids="extract_data", key="targets")

    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42
    )

    ti.xcom_push(key="X_train", value=X_train)
    ti.xcom_push(key="X_test", value=X_test)
    ti.xcom_push(key="y_train", value=y_train)
    ti.xcom_push(key="y_test", value=y_test)
    print(f"แบ่งข้อมูลแล้ว: train {len(X_train)} แถว, test {len(X_test)} แถว")


def train_model(**kwargs):
    """เทรน RandomForestClassifier แล้วเซฟไฟล์โมเดลแบบมี timestamp กำกับ"""
    import joblib
    from sklearn.ensemble import RandomForestClassifier

    ti = kwargs["ti"]
    X_train = ti.xcom_pull(task_ids="prepare_data", key="X_train")
    y_train = ti.xcom_pull(task_ids="prepare_data", key="y_train")

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(MODEL_DIR, exist_ok=True)
    run_id = kwargs["run_id"].replace(":", "-").replace("+", "-")
    candidate_path = os.path.join(MODEL_DIR, f"candidate_{run_id}.pkl")
    joblib.dump(model, candidate_path)

    ti.xcom_push(key="candidate_model_path", value=candidate_path)
    print(f"เทรนโมเดลเสร็จ บันทึกไว้ที่: {candidate_path}")


def evaluate_model(**kwargs):
    """โหลดโมเดลที่เพิ่งเทรน มาวัด accuracy บน test set"""
    import joblib
    from sklearn.metrics import accuracy_score

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")
    X_test = ti.xcom_pull(task_ids="prepare_data", key="X_test")
    y_test = ti.xcom_pull(task_ids="prepare_data", key="y_test")

    model = joblib.load(candidate_path)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    ti.xcom_push(key="accuracy", value=accuracy)
    print(f"ประเมินผลโมเดล: accuracy = {accuracy:.4f} (เกณฑ์ deploy: > {ACCURACY_THRESHOLD})")


def decide_deploy(**kwargs):
    """BranchPythonOperator: ตัดสินใจว่าจะ deploy โมเดลนี้หรือไม่"""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")

    if accuracy > ACCURACY_THRESHOLD:
        print(f"accuracy {accuracy:.4f} > {ACCURACY_THRESHOLD} -> เลือกเส้นทาง deploy_model")
        return "deploy_model"
    else:
        print(f"accuracy {accuracy:.4f} <= {ACCURACY_THRESHOLD} -> เลือกเส้นทาง skip_deploy")
        return "skip_deploy"


def deploy_model(**kwargs):
    """คัดลอกไฟล์โมเดลที่ผ่านเกณฑ์ไปทับ current_model.pkl (จำลอง production)"""
    import shutil

    ti = kwargs["ti"]
    candidate_path = ti.xcom_pull(task_ids="train_model", key="candidate_model_path")

    shutil.copyfile(candidate_path, CURRENT_MODEL_PATH)
    print(f"Deploy สำเร็จ: {candidate_path} -> {CURRENT_MODEL_PATH}")

    return "deployed"


def smoke_test(**kwargs):
    """
    ทดสอบเรียกใช้งานโมเดลที่เพิ่ง deploy จริง ด้วยข้อมูลตัวอย่างไม่กี่แถว
    ทำหน้าที่เป็น 'ด่านสุดท้าย' ยืนยันว่าไฟล์โมเดลใช้งานได้จริง ไม่ใช่แค่
    เขียนไฟล์สำเร็จเฉยๆ (ไฟล์อาจเสียหาย หรือ format ผิดได้ถึงแม้เขียนสำเร็จ)
    """
    import joblib
    from sklearn.datasets import load_iris

    model = joblib.load(CURRENT_MODEL_PATH)
    iris = load_iris()

    sample = iris.data[:3]
    actual = iris.target[:3]
    predicted = model.predict(sample)

    print("===== Smoke Test: เรียกใช้งานโมเดลที่เพิ่ง deploy =====")
    correct_count = 0
    for i in range(3):
        name_actual = iris.target_names[actual[i]]
        name_predicted = iris.target_names[predicted[i]]
        is_correct = actual[i] == predicted[i]
        correct_count += int(is_correct)
        status = "ถูก" if is_correct else "ผิด"
        print(f"แถว {i}: จริง={name_actual}, ทาย={name_predicted} ({status})")

    print(f"สรุป Smoke Test: ทายถูก {correct_count}/3 แถวตัวอย่าง")
    print("โมเดลใช้งานได้จริง พร้อมให้บริการ")


def skip_deploy(**kwargs):
    """ไม่ deploy เพราะโมเดลไม่ผ่านเกณฑ์ (สำหรับ iris มักไม่ค่อยเข้าทางนี้)"""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    print(f"ข้าม deploy: accuracy {accuracy:.4f} ไม่ผ่านเกณฑ์ {ACCURACY_THRESHOLD}")
    print("โมเดลเดิม (current_model.pkl) ยังคงใช้งานต่อไป")

    return "skipped"


def log_result(**kwargs):
    """สรุปผลรอบนี้ ไม่ว่าจะ deploy+smoke test หรือ skip (fan-in จากทั้งสอง branch)"""
    ti = kwargs["ti"]
    accuracy = ti.xcom_pull(task_ids="evaluate_model", key="accuracy")
    skip_result = ti.xcom_pull(task_ids="skip_deploy")

    outcome = "deployed + smoke tested" if skip_result is None else skip_result

    print("===== สรุปผล Stage 1 (iris pipeline) =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ผลลัพธ์: {outcome}")


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="iris_pipeline_dag",
    default_args=default_args,
    description="Workshop 2 Stage 1: ML pipeline พื้นฐานด้วย sklearn iris (deterministic)",
    schedule=None,                  # สั่งรันด้วยมือเท่านั้น เหมาะกับ workshop
    start_date=datetime(2026, 8, 1),
    catchup=False,
    tags=["workshop2", "stage-1", "ml-pipeline"],
) as dag:

    extract_task = PythonOperator(task_id="extract_data", python_callable=extract_data)
    prepare_task = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    decide_task = BranchPythonOperator(task_id="decide_deploy", python_callable=decide_deploy)

    deploy_task = PythonOperator(task_id="deploy_model", python_callable=deploy_model)
    skip_task = PythonOperator(task_id="skip_deploy", python_callable=skip_deploy)

    smoke_test_task = PythonOperator(task_id="smoke_test", python_callable=smoke_test)

    log_task = PythonOperator(
        task_id="log_result",
        python_callable=log_result,
        trigger_rule="none_failed_min_one_success",  # รอแค่ทางใดทางหนึ่งจากสอง branch
    )

    # ลำดับการรัน: extract -> prepare -> train -> evaluate -> decide (branch)
    # -> deploy (แล้วต่อด้วย smoke_test) หรือ skip -> log_result (รวมทั้งสองทางกลับมา)
    extract_task >> prepare_task >> train_task >> evaluate_task >> decide_task
    decide_task >> deploy_task >> smoke_test_task >> log_task
    decide_task >> skip_task >> log_task