import os
import dotenv
import mlflow


dotenv.load_dotenv()

RANDOM_STATE = int(os.environ["RANDOM_STATE"])
MLFLOW_DATA_DIR = os.environ["MLFLOW_DATA_DIR"]
MLFLOW_TRACKING_URI = os.environ["MLFLOW_TRACKING_URI"]

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
