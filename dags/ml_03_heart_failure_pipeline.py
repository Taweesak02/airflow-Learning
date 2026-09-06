"""
ml_03_heart_failure_pipeline.py
================
Workshop 2: ML Pipeline บน Apache Airflow (Stage 3 - Heart Failure classification)

Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
Target: HeartDisease (0 = Normal, 1 = heart disease), ไม่ใช่เวลารอดชีวิต

โครงสร้าง task จัดใหม่ให้ตรงกับ ml_02_weather_pipeline.py (champion-challenger):

  create_tables -> extract_data -> prepare_data -> train_model -> evaluate_model
  -> get_previous_metrics -> decide_deploy (branch)
       |-> deploy_model -> smoke_test --|
       |-> skip_deploy ----------------|--> log_result -> predict_data

ต่างจาก ml_02 ตรงที่ไฟล์นี้เป็นงาน classification จึงใช้ F1 เป็นเกณฑ์เทียบ
champion (ยิ่งสูงยิ่งดี) แทน RMSE (ยิ่งต่ำยิ่งดี) และมีเกณฑ์ขั้นต่ำ MIN_F1
เป็นพื้นอีกชั้นหนึ่ง กันไม่ให้โมเดลแย่ ๆ ขึ้น production ตอนที่ยังไม่มี champion

*** ข้อกำหนดก่อนรันไฟล์นี้ ***
- ต้องมี Airflow Connection "postgres_target" (ตัวเดียวกับ workshop 1 stage 5
  และ ml_02 — host: postgres_target, port: 5432) ใช้เก็บประวัติ metrics ทุกรอบ
- ต้อง mount โฟลเดอร์ ./models ไว้แล้ว (มีอยู่แล้วใน docker-compose)

วิธีใช้:
1. Trigger DAG heart_failure_pipeline_dag ใน Airflow
2. ถ้า Kaggle ไม่อนุญาตดาวน์โหลดอัตโนมัติ ให้ดาวน์โหลดและแตก heart.csv
   วางที่ ./dags/data/heart.csv แล้วรันใหม่
3. ผลของ "ทุกรอบ" อยู่ที่ ./models/heart_failure_models/<run folder>/
   (evaluation.json, test_predictions.csv, model.joblib)
4. เฉพาะรอบที่ผ่านด่าน decide_deploy เท่านั้นที่จะถูกคัดลอกไปที่
   ./models/heart_failure_models/current/ ซึ่งเป็นตัวที่ model_service เสิร์ฟจริง
5. ทำนายข้อมูลใหม่: วาง CSV ที่มี 11 features เหมือน heart.csv (ไม่ต้องมี target)
   ที่ ./dags/data/heart_predict.csv แล้ว Trigger พร้อม configuration:
   {"prediction_csv": "/opt/airflow/dags/data/heart_predict.csv"}

ใช้ pandas, scikit-learn, joblib, requests และ models volume จาก compose เดิม
โมเดลบันทึก preprocessing รวมไว้แล้ว: model.predict(dataframe) ใช้งานได้ทันที
คะแนนจาก test set ใช้รายงานผล ไม่ได้ใช้เลือกโมเดลหรือปรับ hyperparameters
"""

import hashlib
import io
import json
import math
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile, ZipFile

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

POSTGRES_CONN_ID = "postgres_target"   # connection เดียวกับ workshop 1 stage 5 / ml_02
MODEL_NAME = "heart_failure_rf"

DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"
MODEL_DIR = Path("/opt/airflow/models/heart_failure_models")
# โฟลเดอร์ของโมเดลที่ "ผ่านด่าน" แล้ว เทียบเท่า current_model.pkl ของ ml_02
CURRENT_DIR = MODEL_DIR / "current"
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"

MIN_F1 = 0.75   # เกณฑ์ขั้นต่ำ ต้องผ่านก่อนเสมอ แม้ยังไม่มี champion ให้เทียบ

NUMERIC = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
CATEGORICAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "HeartDisease"
CATEGORIES = {
    "Sex": {"M", "F"},
    "ChestPainType": {"TA", "ATA", "NAP", "ASY"},
    "RestingECG": {"Normal", "ST", "LVH"},
    "ExerciseAngina": {"Y", "N"},
    "ST_Slope": {"Up", "Flat", "Down"},
}

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS heart_model_metrics (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    run_dir VARCHAR(200) NOT NULL,
    accuracy FLOAT NOT NULL,
    precision_score FLOAT NOT NULL,
    recall_score FLOAT NOT NULL,
    f1 FLOAT NOT NULL,
    roc_auc FLOAT NOT NULL,
    deployed BOOLEAN NOT NULL,
    run_at TIMESTAMP DEFAULT NOW()
);
"""


def validate_features(frame):
    """ตรวจ schema และทำความสะอาดแบบไม่เรียนรู้สถิติจากข้อมูล."""
    import numpy as np
    import pandas as pd

    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"CSV ขาดคอลัมน์: {missing}")
    if frame.empty:
        raise ValueError("CSV ไม่มีข้อมูล")
    result = frame[FEATURES].copy()
    for column in NUMERIC:
        result[column] = pd.to_numeric(result[column], errors="raise")
        if np.isinf(result[column].to_numpy(dtype=float)).any():
            raise ValueError(f"{column} มีค่า infinity")
    for column, allowed in CATEGORIES.items():
        result[column] = result[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        ).replace("", np.nan)
        if not result[column].dropna().isin(allowed).all():
            raise ValueError(f"{column} ต้องอยู่ใน {sorted(allowed)}")
    if not result["FastingBS"].dropna().isin([0, 1]).all():
        raise ValueError("FastingBS ต้องเป็น 0 หรือ 1")
    # ค่า 0 ในสองคอลัมน์นี้แทนค่าที่ขาด; median จะ fit เฉพาะ train set
    result[["RestingBP", "Cholesterol"]] = result[["RestingBP", "Cholesterol"]].replace(0, np.nan)
    return result


def extract_data(**context):
    import pandas as pd
    import requests

    run_key = hashlib.sha256(context["run_id"].encode()).hexdigest()[:20]
    run_dir = MODEL_DIR / run_key
    run_dir.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.is_file():
        raw = DATA_PATH.read_bytes()
        print(f"Reading local dataset: {DATA_PATH}")
    else:
        try:
            response = requests.get(DOWNLOAD_URL, timeout=(15, 120))
            response.raise_for_status()
            with ZipFile(io.BytesIO(response.content)) as archive:
                matches = [name for name in archive.namelist() if Path(name).name == "heart.csv"]
                if len(matches) != 1:
                    raise ValueError("ไม่พบ heart.csv เพียงไฟล์เดียวใน Kaggle archive")
                raw = archive.read(matches[0])
        except (requests.RequestException, BadZipFile, ValueError) as exc:
            raise RuntimeError(
                f"Kaggle download failed ({type(exc).__name__}: {exc}). "
                f"Place heart.csv at {DATA_PATH} (host: ./dags/data/heart.csv), "
                "then clear extract_data and its downstream tasks, or trigger a new run."
            ) from exc
    frame = pd.read_csv(io.BytesIO(raw))
    features = validate_features(frame)
    if TARGET not in frame or frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("ต้องมี HeartDisease เป็น 0 หรือ 1 ทุกแถว")
    features[TARGET] = frame[TARGET].astype(int)
    features = features.drop_duplicates().reset_index(drop=True)
    if len(features) < 20 or features[TARGET].value_counts().reindex([0, 1], fill_value=0).min() < 5:
        raise ValueError("ต้องมีข้อมูลอย่างน้อย 20 แถว และแต่ละ class อย่างน้อย 5 แถว")
    features.to_csv(run_dir / "dataset.csv", index=False)
    (run_dir / "source.json").write_text(json.dumps({
        "source": str(DATA_PATH) if DATA_PATH.is_file() else DOWNLOAD_URL,
        "sha256": hashlib.sha256(raw).hexdigest(), "run_id": context["run_id"],
        "rows_after_deduplication": len(features),
    }, indent=2), encoding="utf-8")
    return str(run_dir)


def prepare_data(**context):
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split

    run_dir = Path(context["ti"].xcom_pull(task_ids="extract_data"))
    frame = pd.read_csv(run_dir / "dataset.csv")
    split = train_test_split(frame[FEATURES], frame[TARGET], test_size=0.2,
                             stratify=frame[TARGET], random_state=42)
    if split[0].isna().all().any():
        raise ValueError("Train set มีคอลัมน์ที่ไม่มีค่าทั้งคอลัมน์")
    joblib.dump(split, run_dir / "split.joblib")
    return str(run_dir)


def train_model(**context):
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    run_dir = Path(context["ti"].xcom_pull(task_ids="prepare_data"))
    X_train, _, y_train, _ = joblib.load(run_dir / "split.joblib")
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight="balanced", random_state=42, n_jobs=1)),
    ])
    model.fit(X_train, y_train)
    joblib.dump(model, run_dir / "model.joblib")
    return str(run_dir)


def evaluate_model(**context):
    """วัดผลบน test set ที่กันไว้ แล้วเขียน evaluation.json ของรอบนี้

    ตั้งใจ "ไม่" เขียน metrics.json ที่นี่ เพราะ model_service ใช้การมีอยู่ของ
    metrics.json เป็นสัญญาณว่าโมเดลพร้อมเสิร์ฟ — ไฟล์นั้นจึงเป็นหน้าที่ของ
    deploy_model เท่านั้น (ดูคอมเมนต์ใน deploy_model)
    """
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="train_model"))
    X_train, X_test, _, y_test = joblib.load(run_dir / "split.joblib")
    model = joblib.load(run_dir / "model.joblib")
    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": [0, 1],
        "train_rows": len(X_train), "test_rows": len(X_test),
        "sklearn_version": sklearn.__version__, "random_state": 42,
    }
    (run_dir / "evaluation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output = X_test.copy()
    output["actual_HeartDisease"] = y_test
    output["predicted_HeartDisease"] = predicted
    output["probability_HeartDisease"] = probability
    output.to_csv(run_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    ti.xcom_push(key="f1", value=metrics["f1"])
    return metrics


def get_previous_metrics(**context):
    """ดึง F1 ของโมเดลที่ deploy ล่าสุดจาก Postgres มาเทียบ (champion)"""
    ti = context["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    row = hook.get_first(
        "SELECT f1 FROM heart_model_metrics "
        "WHERE model_name = %s AND deployed = TRUE "
        "ORDER BY run_at DESC LIMIT 1;",
        parameters=(MODEL_NAME,),
    )
    previous_f1 = float(row[0]) if row else None

    ti.xcom_push(key="previous_f1", value=previous_f1)
    if previous_f1 is None:
        print("ยังไม่เคยมีโมเดลที่ deploy มาก่อน (รอบแรก) — ยังไม่มี champion ให้เทียบ")
    else:
        print(f"F1 ของโมเดลที่ deploy ล่าสุด (champion ปัจจุบัน): {previous_f1:.4f}")


def decide_deploy(**context):
    """BranchPythonOperator: deploy เมื่อผ่านเกณฑ์ขั้นต่ำ และดีกว่า champion เดิม"""
    ti = context["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1")
    previous_f1 = ti.xcom_pull(task_ids="get_previous_metrics", key="previous_f1")

    if f1 < MIN_F1:
        print(f"F1 {f1:.4f} ต่ำกว่าเกณฑ์ขั้นต่ำ {MIN_F1} -> skip_deploy")
        return "skip_deploy"
    if previous_f1 is None:
        print(f"F1 {f1:.4f} ผ่านเกณฑ์ขั้นต่ำ และยังไม่มี champion -> deploy_model")
        return "deploy_model"
    if f1 > previous_f1:
        print(f"F1 ใหม่ {f1:.4f} ดีกว่า champion เดิม {previous_f1:.4f} -> deploy_model")
        return "deploy_model"
    print(f"F1 ใหม่ {f1:.4f} ไม่ดีกว่า champion เดิม {previous_f1:.4f} -> skip_deploy")
    return "skip_deploy"


def deploy_model(**context):
    """คัดลอกโมเดลที่ชนะไปที่โฟลเดอร์ current/ (จำลอง production)

    model_service เลือกโมเดลจาก glob("*/model.joblib") ที่ "มี metrics.json อยู่ข้าง ๆ"
    การเขียน metrics.json จึงเท่ากับการประกาศว่าโมเดลตัวนี้พร้อมเสิร์ฟ — ทำที่นี่
    ที่เดียว รอบที่ไม่ผ่านด่านจะไม่มี metrics.json จึงไม่ถูกหยิบไปเสิร์ฟ
    """
    ti = context["ti"]
    run_dir = Path(ti.xcom_pull(task_ids="train_model"))
    CURRENT_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(run_dir / "model.joblib", CURRENT_DIR / "model.joblib")
    shutil.copyfile(run_dir / "evaluation.json", CURRENT_DIR / "metrics.json")
    (CURRENT_DIR / "promoted_from.json").write_text(json.dumps({
        "run_dir": run_dir.name,
        "run_id": context["run_id"],
        "promoted_at": datetime.now().isoformat(),
    }, indent=2), encoding="utf-8")

    # ทำเครื่องหมายที่ run dir ต้นทางด้วย เพื่อให้ย้อนดูได้ว่ารอบไหนเคยขึ้น production
    shutil.copyfile(run_dir / "evaluation.json", run_dir / "metrics.json")

    print(f"Deploy สำเร็จ: {run_dir} -> {CURRENT_DIR}")
    return "deployed"


def skip_deploy(**context):
    """ไม่ deploy เพราะยังสู้โมเดลเดิม (champion) ไม่ได้ หรือไม่ผ่านเกณฑ์ขั้นต่ำ"""
    ti = context["ti"]
    f1 = ti.xcom_pull(task_ids="evaluate_model", key="f1")
    previous_f1 = ti.xcom_pull(task_ids="get_previous_metrics", key="previous_f1")
    champion = "ยังไม่มี" if previous_f1 is None else f"{previous_f1:.4f}"
    print(f"ข้าม deploy: F1 ใหม่ {f1:.4f} vs champion เดิม {champion} (เกณฑ์ขั้นต่ำ {MIN_F1})")
    print(f"โมเดลเดิมใน {CURRENT_DIR} ยังคงใช้งานต่อไป")
    return "skipped"


def smoke_test(**context):
    """ทดสอบเรียกใช้งานโมเดลที่เพิ่ง deploy จริง ด้วยข้อมูลตัวอย่างไม่กี่แถว

    ทำหน้าที่เป็น 'ด่านสุดท้าย' ยืนยันว่าไฟล์โมเดลใน current/ ใช้งานได้จริง
    ไม่ใช่แค่คัดลอกไฟล์สำเร็จเฉย ๆ (ไฟล์อาจเสียหายระหว่างคัดลอกได้)
    """
    import joblib

    run_dir = Path(context["ti"].xcom_pull(task_ids="train_model"))
    current_model_path = CURRENT_DIR / "model.joblib"

    # ด่าน 1: ไฟล์ที่คัดลอกไป current/ ต้องเหมือนต้นฉบับ byte ต่อ byte
    # (deploy_model ใช้ shutil.copyfile — ถ้าดิสก์เต็มหรือคัดลอกค้างจะจับได้ตรงนี้)
    source_hash = hashlib.sha256((run_dir / "model.joblib").read_bytes()).hexdigest()
    deployed_hash = hashlib.sha256(current_model_path.read_bytes()).hexdigest()
    if source_hash != deployed_hash:
        raise ValueError(
            f"Smoke test ไม่ผ่าน: ไฟล์โมเดลใน current/ ไม่ตรงกับต้นฉบับใน {run_dir.name} "
            f"(sha256 {deployed_hash[:12]} != {source_hash[:12]})"
        )

    # ด่าน 2: ต้องมี metrics.json อยู่ข้าง ๆ ไม่งั้น model_service จะมองไม่เห็นโมเดลนี้
    metrics_path = CURRENT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Smoke test ไม่ผ่าน: ไม่พบ {metrics_path} — model_service จะไม่หยิบโมเดลนี้ไปเสิร์ฟ"
        )
    json.loads(metrics_path.read_text(encoding="utf-8"))

    model = joblib.load(current_model_path)
    _, X_test, _, y_test = joblib.load(run_dir / "split.joblib")

    samples = X_test.head(5).copy()
    if samples.empty:
        raise ValueError("Smoke test ไม่ผ่าน: ไม่มีข้อมูล test เหลือให้ทดสอบเลย")
    actual = y_test.head(5).tolist()
    predicted = model.predict(samples)
    probability = model.predict_proba(samples)[:, list(model.classes_).index(1)]

    print("===== Smoke Test: เรียกใช้งานโมเดลที่เพิ่ง deploy =====")
    correct_count = 0
    for i in range(len(samples)):
        is_correct = actual[i] == predicted[i]
        correct_count += int(is_correct)
        status = "ถูก" if is_correct else "ผิด"
        print(f"แถว {i}: จริง={actual[i]}, ทาย={predicted[i]} "
              f"(ความน่าจะเป็น {probability[i]:.3f}) ({status})")

    output = samples.copy()
    output["actual_HeartDisease"] = actual
    output["predicted_HeartDisease"] = predicted
    output["probability_HeartDisease"] = probability
    output.to_csv(run_dir / "sample_predictions.csv", index=False)

    # ด่าน 3: ผลลัพธ์ต้องอยู่ในรูปที่ใช้งานได้จริง ไม่ใช่แค่ "เรียกแล้วไม่ error"
    valid_labels = set(model.classes_)
    unexpected = set(predicted) - valid_labels
    if unexpected:
        raise ValueError(f"Smoke test ไม่ผ่าน: โมเดลทายค่าที่ไม่รู้จัก {unexpected}")
    for i, prob in enumerate(probability):
        if not math.isfinite(prob) or not (0.0 <= prob <= 1.0):
            raise ValueError(
                f"Smoke test ไม่ผ่าน: ความน่าจะเป็นแถว {i} ผิดปกติ ({prob}) — ต้องอยู่ระหว่าง 0-1"
            )

    print(f"สรุป Smoke Test: ทายถูก {correct_count}/{len(samples)} แถวตัวอย่าง")
    if correct_count == 0:
        # ไม่ raise เพราะ 5 แถวน้อยเกินกว่าจะตัดสิน — แต่ต้องเห็นชัดใน log
        print("เตือน: ทายผิดทั้ง 5 แถว ควรไปดู evaluate_model ว่า metrics ตรงกันไหม")
    print("โมเดลใช้งานได้จริง พร้อมให้บริการ")


def log_result(**context):
    """บันทึก metrics รอบนี้ลง Postgres (ให้รอบถัดไปดึงไปเทียบเป็น champion)"""
    ti = context["ti"]
    hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)

    metrics = ti.xcom_pull(task_ids="evaluate_model")
    run_dir = Path(ti.xcom_pull(task_ids="train_model"))
    skip_result = ti.xcom_pull(task_ids="skip_deploy")
    deployed = skip_result is None  # ถ้า skip_deploy ไม่ได้รัน แปลว่า deploy ไปแล้ว

    hook.run(
        "INSERT INTO heart_model_metrics "
        "(model_name, run_dir, accuracy, precision_score, recall_score, f1, roc_auc, deployed) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s);",
        parameters=(MODEL_NAME, run_dir.name, metrics["accuracy"], metrics["precision"],
                    metrics["recall"], metrics["f1"], metrics["roc_auc"], deployed),
    )

    print("===== สรุปผล Stage 3 (heart failure pipeline) =====")
    print(f"run dir: {run_dir.name}")
    print(f"accuracy: {metrics['accuracy']:.4f} | f1: {metrics['f1']:.4f} | "
          f"roc_auc: {metrics['roc_auc']:.4f}")
    print(f"Deploy รอบนี้: {'ใช่' if deployed else 'ไม่ใช่'}")


def predict_data(**context):
    """Batch inference ด้วยโมเดลที่เสิร์ฟอยู่จริง (current/) — ทำเฉพาะเมื่อส่ง conf มา

    Trigger พร้อม configuration: {"prediction_csv": "/opt/airflow/dags/data/heart_predict.csv"}
    ถ้าไม่ส่งมาจะข้าม task นี้ไปเฉย ๆ (การทดสอบโหลดโมเดลย้ายไปอยู่ที่ smoke_test แล้ว)
    """
    import joblib
    import pandas as pd

    prediction_csv = (context["dag_run"].conf or {}).get("prediction_csv")
    if not prediction_csv:
        print("ไม่ได้ส่ง prediction_csv มาใน configuration — ข้าม batch inference")
        return None

    model_path = CURRENT_DIR / "model.joblib"
    if not model_path.is_file():
        raise RuntimeError(
            f"ยังไม่มีโมเดลที่ผ่านด่าน deploy ใน {CURRENT_DIR} "
            "(รอบนี้อาจเข้าทาง skip_deploy) จึงยังทำนายไม่ได้"
        )

    run_dir = Path(context["ti"].xcom_pull(task_ids="train_model"))
    model = joblib.load(model_path)
    samples = validate_features(pd.read_csv(prediction_csv))
    output = samples.copy()
    output["predicted_HeartDisease"] = model.predict(samples)
    output["probability_HeartDisease"] = model.predict_proba(samples)[:, list(model.classes_).index(1)]
    path = run_dir / "predictions.csv"
    output.to_csv(path, index=False)
    print(f"บันทึกผลทำนาย {len(output)} แถว: {path}")
    return str(path)


# -----------------------------------------------------------------
# นิยาม DAG
# -----------------------------------------------------------------
with DAG(
    dag_id="heart_failure_pipeline_dag",
    default_args=default_args,
    description="Workshop 2 Stage 3: Kaggle heart disease ด้วย champion-challenger บน Postgres",
    schedule=None,
    start_date=datetime(2026, 8, 1),
    catchup=False,
    max_active_runs=1,
    tags=["workshop2", "stage-3", "ml-pipeline", "heart-failure"],
    doc_md=__doc__,
) as dag:

    create_tables_task = PostgresOperator(
        task_id="create_tables",
        postgres_conn_id=POSTGRES_CONN_ID,
        sql=CREATE_TABLES_SQL,
    )

    extract_task = PythonOperator(task_id="extract_data", python_callable=extract_data)
    prepare_task = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    train_task = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate_task = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)

    previous_metrics_task = PythonOperator(
        task_id="get_previous_metrics",
        python_callable=get_previous_metrics,
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

    predict_task = PythonOperator(task_id="predict_data", python_callable=predict_data)

    # ลำดับการรันทั้งหมด (รูปทรงเดียวกับ ml_02: เส้นตรง -> branch -> มาบรรจบที่ log_result)
    (
        create_tables_task
        >> extract_task
        >> prepare_task
        >> train_task
        >> evaluate_task
        >> previous_metrics_task
        >> decide_task
    )
    decide_task >> deploy_task >> smoke_test_task >> log_task
    decide_task >> skip_task >> log_task
    log_task >> predict_task
