# Import Path for safe file handling.
from pathlib import Path


# Define the project root directory.
BASE_DIR = Path(__file__).resolve().parent.parent

# Define the data directory path.
DATA_DIR = BASE_DIR / "data"

# Define the artifacts directory path.
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Define the logs directory path.
LOGS_DIR = BASE_DIR / "logs"

# Define the input dataset path.
DATA_PATH = DATA_DIR / "final.csv"

# Define the model output path.
MODEL_PATH = ARTIFACTS_DIR / "random_forest_model.joblib"

# Define the feature columns output path.
FEATURES_PATH = ARTIFACTS_DIR / "feature_columns.json"

# Define the metrics output path.
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"