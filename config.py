from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "Plant_Disease_Model.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "plant_leaf_class_names.json"
HISTORY_PATH = PROJECT_ROOT / "data" / "prediction_history.csv"

MODEL_INPUT_SIZE = (224, 224)
MODEL_INPUT_SHAPE = (224, 224, 3)
EXPECTED_CLASS_COUNT = 38
TOP_PREDICTIONS = 3

ALLOWED_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")
