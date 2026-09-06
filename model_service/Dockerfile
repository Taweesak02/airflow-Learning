# Dockerfile — Iris Model Serving API (Workshop 2, ตัวอย่างขั้นสูง)
# แยก image ออกจาก Airflow โดยสิ้นเชิง เบาและเริ่มเร็วกว่ามาก
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py weather.py heart_failure.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
