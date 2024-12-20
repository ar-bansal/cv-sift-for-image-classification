from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from ..mlflow_utils.mlflow_utils import mlflow_log


@mlflow_log
def run_clustering_pipeline(X_train, y_train, pipeline: Pipeline, model_params, experiment_name: str):
    if model_params is not None:
        pipeline.set_params(model_params)
    pipeline.fit(X_train, y_train)
    predicted_labels = pipeline.predict(X_train)
    silhouette = silhouette_score(X_train, predicted_labels)

    metrics = {
        "silhouette_score": silhouette
    }

    return pipeline, metrics