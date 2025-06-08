from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
import pickle
import mlflow
import os
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.sparse import vstack, save_npz, load_npz
from uuid import uuid4

default_args = {
    'start_date': datetime(2023, 3, 1)
}

DATA_DIR = '/opt/airflow/data'

with DAG("nyc_taxi_pipeline",
         schedule_interval=None,
         default_args=default_args,
         catchup=False) as dag:

    def download_data():
        DATA_DIR = "/opt/airflow/data"
        URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
        OUTPUT_PATH = f"{DATA_DIR}/raw.parquet"
        os.makedirs(DATA_DIR, exist_ok=True)

        # Get remote file size first
        head = requests.head(URL)
        total_size = int(head.headers.get('content-length', 0))

        existing_size = os.path.getsize(OUTPUT_PATH) if os.path.exists(OUTPUT_PATH) else 0
        if existing_size >= total_size:
            print("File already fully downloaded.")
            return

        headers = {}
        mode = "wb"
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"

        # Configure retries
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504]
        )
        session.mount("https://", HTTPAdapter(max_retries=retries))

        with session.get(URL, stream=True, headers=headers, timeout=60) as r:
            r.raise_for_status()
            with open(OUTPUT_PATH, mode) as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")


    def prepare_data():
        cleaned_path = f"{DATA_DIR}/cleaned.parquet"
        if os.path.exists(cleaned_path):
            print("Cleaned data already exists. Skipping preparation.")
            return

        cols = ["PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
        df = pd.read_parquet(f"{DATA_DIR}/raw.parquet", columns=cols)
        print("Q3 - Loaded records:", len(df))
        
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
        df = df[(df.duration >= 1) & (df.duration <= 60)]
        df[['PULocationID', 'DOLocationID']] = df[['PULocationID', 'DOLocationID']].astype(str)
        
        print(f"Q4 - Filtered records: {len(df)}")
        df.to_parquet(cleaned_path)
        print("Saved cleaned data.")

    def train_model():
        df = pd.read_parquet(f"{DATA_DIR}/cleaned.parquet")
        categorical = ['PULocationID', 'DOLocationID']
        y = df['duration'].values

        dv = DictVectorizer()
        unique_df = df[categorical].drop_duplicates()
        dv.fit(unique_df.to_dict(orient='records'))

        batch_size = 250_000
        matrix_paths = []
        y_batches = []

        print("Starting transformation & batching...")
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            dicts = batch[categorical].to_dict(orient='records')
            X_batch = dv.transform(dicts)
            y_batch = batch['duration'].values

            # Save batch matrix and target to disk
            batch_id = str(uuid4())
            matrix_path = f"/tmp/X_{batch_id}.npz"
            save_npz(matrix_path, X_batch)
            matrix_paths.append(matrix_path)
            y_batches.append(y_batch)

        # Now load batches and stack them
        print("Loading batches from disk...")
        X_all = vstack([load_npz(path) for path in matrix_paths])
        y_all = pd.concat([pd.Series(yb) for yb in y_batches]).values

        print("Training LinearRegression model...")
        model = LinearRegression()
        model.fit(X_all, y_all)

        with open(f"{DATA_DIR}/model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open(f"{DATA_DIR}/dv.pkl", "wb") as f:
            pickle.dump(dv, f)

        print(f"Q5 - Intercept: {model.intercept_:.2f}, Coeffs: {model.coef_.shape}")

    def log_model():
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("nyc-taxi")

        with open(f"{DATA_DIR}/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{DATA_DIR}/dv.pkl", "rb") as f:
            dv = pickle.load(f)

        with mlflow.start_run():
            mlflow.sklearn.log_model(model, artifact_path="model")
            dv_path = "/tmp/dv.pkl"
            with open(dv_path, "wb") as f:
                pickle.dump(dv, f)
            mlflow.log_artifact(dv_path, artifact_path="preprocessor")
        print("Q6 - Model logged to MLflow.")

    t1 = PythonOperator(task_id='download_data', python_callable=download_data)
    t2 = PythonOperator(task_id='prepare_data', python_callable=prepare_data)
    t3 = PythonOperator(task_id='train_model', python_callable=train_model)
    t4 = PythonOperator(task_id='log_model', python_callable=log_model)

    t1 >> t2 >> t3 >> t4