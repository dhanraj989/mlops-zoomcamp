import os
import pickle
import click
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        new_params = {}
        for param in RF_PARAMS:
            new_params[param] = int(params[param])

        rf = RandomForestRegressor(**new_params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)

        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric("test_rmse", test_rmse)

        # Explicitly log the model
        mlflow.sklearn.log_model(rf, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models that need to be evaluated to decide which one to promote"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    try:
        # Retrieve HPO experiment and top N runs
        print("Fetching top runs from HPO experiment...")
        hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
        hpo_runs = client.search_runs(
            experiment_ids=hpo_experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=top_n,
            order_by=["metrics.rmse ASC"]
        )

        print(f"Retrieved {len(hpo_runs)} runs. Evaluating...")

        for run in hpo_runs:
            print(f"Evaluating run_id: {run.info.run_id}")
            print("Params:", run.data.params)
            train_and_log_model(data_path=data_path, params=run.data.params)

        print("Fetching best run from best-models experiment...")
        best_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        best_runs = client.search_runs(
            experiment_ids=best_experiment.experiment_id,
            run_view_type=ViewType.ACTIVE_ONLY,
            order_by=["metrics.test_rmse ASC"]
        )

        best_run = best_runs[0]
        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"

        print(f"Registering model from run {run_id}")
        mlflow.register_model(model_uri=model_uri, name="random-forest-best-model")

        print("Model registered successfully.")

    except Exception as e:
        print(f"❌ Error occurred: {e}")

if __name__ == '__main__':
    run_register_model()
