FROM python:3.10-slim

RUN pip install --upgrade pip && \
    pip install \
    --default-timeout=100 \
    --retries=10 \
    --resume-retries=10 \
    --no-cache-dir \
    mlflow[extras]==2.12.1

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///home/mlflow_data/mlflow.db", "--host", "0.0.0.0", "--port", "5000"]
