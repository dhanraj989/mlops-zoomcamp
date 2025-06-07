if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def export(data, *args, **kwargs):
    import mlflow
    import os
    import pickle

    dv, model = data

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("nyc-taxi")

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, artifact_path="model")

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/dv.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("artifacts/dv.pkl", artifact_path="preprocessor")