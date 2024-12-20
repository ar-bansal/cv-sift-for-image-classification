import os
import mlflow
import mlflow.sklearn
from functools import wraps
from ..config.config import MLFLOW_DATA_DIR


__all__ = ["mlflow_log"]


# MLFLOW_DATA_DIR = os.environ["MLFLOW_DATA_DIR"]


def get_experiment_id(experiment_name: str):
    """
    Retrieve the experiment ID for the experiment name. Create 
    a new experiment if it does not exist.

    Parameters:
        - experiment_name (str): The MLFlow experiment name.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
    except AttributeError:
        artifact_location = os.path.join(
            MLFLOW_DATA_DIR, 
            experiment_name
        )
        experiment_id = mlflow.create_experiment(experiment_name, artifact_location=artifact_location)

    return experiment_id


def mlflow_log(func):
    """
    Decorator for logging model parameters, metrics, and the model artifact to MLflow.

    Parameters: 
        - experiment_name (str): The MLFlow experiment name.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Set the experiment
        experiment_name = kwargs["experiment_name"]
        experiment_id = get_experiment_id(experiment_name)
        mlflow.set_experiment(experiment_id=experiment_id)

        mlflow.sklearn.autolog(serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)
        with mlflow.start_run():
            model, metrics = func(*args, **kwargs)
            for metric_name, metric_val in metrics.items():
                if "confusion_matrix" in metric_name:
                    metric_val.to_csv(metric_name + ".csv", index=False)
                    mlflow.log_artifact(metric_name + ".csv")
                else:
                    mlflow.log_metric(metric_name, metric_val)

        return model, metrics
    return wrapper