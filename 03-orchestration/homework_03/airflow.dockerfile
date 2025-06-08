FROM apache/airflow:2.9.1

# Upgrade pip and install necessary Python packages with robust retry logic
RUN pip install --upgrade pip && \
    pip install \
    --default-timeout=100 \
    --retries=10 \
    --resume-retries=3 \
    --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    pandas scikit-learn mlflow[extras]==2.12.1
