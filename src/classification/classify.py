import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from ..mlflow_utils.mlflow_utils import mlflow_log


@mlflow_log
def run_classification_pipeline(
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    pipeline: Pipeline, 
    model_params, 
    experiment_name: str
):
    if experiment_name is None:
        raise ValueError("Please enter a valid experiment name")
    
    # Training the model
    if model_params is not None:
        pipeline.set_params(model_params)
    pipeline.fit(X_train, y_train)

    cm_labels = np.unique(y_train).tolist()
    true_cols = [f"true_{l}" for l in cm_labels]
    pred_idxs = [f"pred_{l}" for l in cm_labels]

    # Training metrics
    train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    train_cm = confusion_matrix(y_train, train_pred, labels=cm_labels).T

    # Validation metrics
    val_pred = pipeline.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_cm = confusion_matrix(y_val, val_pred, labels=cm_labels).T

    metrics = {
        "train_acccuracy": train_accuracy, 
        "train_confusion_matrix": pd.DataFrame(train_cm, columns=true_cols, index=pred_idxs), 
        "validation_accuracy": val_accuracy, 
        "validation_confusion_matrix": pd.DataFrame(val_cm, columns=true_cols, index=pred_idxs)
    }

    return pipeline, metrics