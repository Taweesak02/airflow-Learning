"""Airflow ML pipeline สำหรับ Heart Failure Prediction Dataset.

Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data
Target: HeartDisease (0 = Normal, 1 = heart disease), ไม่ใช่เวลารอดชีวิต

วิธีใช้:
1. Trigger DAG heart_failure_pipeline_dag ใน Airflow
2. ถ้า Kaggle ไม่อนุญาตดาวน์โหลดอัตโนมัติ ให้ดาวน์โหลดและแตก heart.csv
   วางที่ ./dags/data/heart.csv แล้วรันใหม่
3. ดู metrics.json, test_predictions.csv, sample_predictions.csv และ model.joblib
   ใน ./models/heart_failure_models/<run folder>/
4. ทำนายข้อมูลใหม่: วาง CSV ที่มี 11 features เหมือน heart.csv (ไม่ต้องมี target)
   ที่ ./dags/data/heart_predict.csv แล้ว Trigger พร้อม configuration:
   {"prediction_csv": "/opt/airflow/dags/data/heart_predict.csv"}

ใช้ pandas, scikit-learn, joblib, requests และ models volume จาก compose เดิม
โมเดลบันทึก preprocessing รวมไว้แล้ว: model.predict(dataframe) ใช้งานได้ทันที
คะแนนจาก test set ใช้รายงานผล ไม่ได้ใช้เลือกโมเดลหรือปรับ hyperparameters
"""

import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_PATH = Path(__file__).resolve().parent / "data" / "heart.csv"
MODEL_DIR = Path("/opt/airflow/models/heart_failure_models")
DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/fedesoriano/heart-failure-prediction"
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
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

    run_dir = Path(context["ti"].xcom_pull(task_ids="train_model"))
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
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    output = X_test.copy()
    output["actual_HeartDisease"] = y_test
    output["predicted_HeartDisease"] = predicted
    output["probability_HeartDisease"] = probability
    output.to_csv(run_dir / "test_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    return metrics


def predict_data(**context):
    import joblib
    import pandas as pd

    run_dir = Path(context["ti"].xcom_pull(task_ids="train_model"))
    model = joblib.load(run_dir / "model.joblib")
    prediction_csv = (context["dag_run"].conf or {}).get("prediction_csv")
    if prediction_csv:
        samples = validate_features(pd.read_csv(prediction_csv))
        filename = "predictions.csv"
    else:
        # ตัวอย่างจาก holdout เพื่อทดสอบโหลดโมเดลกลับมาใช้งาน
        _, X_test, _, _ = joblib.load(run_dir / "split.joblib")
        samples = X_test.head(5).copy()
        filename = "sample_predictions.csv"
    output = samples.copy()
    output["predicted_HeartDisease"] = model.predict(samples)
    output["probability_HeartDisease"] = model.predict_proba(samples)[:, list(model.classes_).index(1)]
    path = run_dir / filename
    output.to_csv(path, index=False)
    print(f"บันทึกผลทำนาย {len(output)} แถว: {path}")
    return str(path)


with DAG(
    dag_id="heart_failure_pipeline_dag",
    description="Kaggle heart disease: extract, prepare, train, evaluate, predict",
    default_args={"owner": "workshop2", "retries": 1, "retry_delay": timedelta(minutes=2)},
    start_date=datetime(2026, 8, 1), schedule=None, catchup=False,
    max_active_runs=1, tags=["workshop2", "ml-pipeline", "heart-failure"],
    doc_md=__doc__,
) as dag:
    extract = PythonOperator(task_id="extract_data", python_callable=extract_data)
    prepare = PythonOperator(task_id="prepare_data", python_callable=prepare_data)
    train = PythonOperator(task_id="train_model", python_callable=train_model)
    evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    predict = PythonOperator(task_id="predict_data", python_callable=predict_data)
    extract >> prepare >> train >> evaluate >> predict
