from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import keras
import numpy as np

from config import (
    CLASS_NAMES_PATH,
    EXPECTED_CLASS_COUNT,
    MODEL_INPUT_SHAPE,
    MODEL_PATH,
)


class PredictionError(RuntimeError):
    """Raised when model loading or prediction is not valid."""


@lru_cache(maxsize=1)
def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained Keras model once and validate its input/output shapes."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise PredictionError(f"Model file was not found: {model_path}")

    try:
        model = keras.models.load_model(model_path, compile=False)
    except Exception as exc:
        raise PredictionError(f"The trained model could not be loaded: {exc}") from exc

    if tuple(model.input_shape) != (None, *MODEL_INPUT_SHAPE):
        raise PredictionError(
            f"Unexpected model input shape: {model.input_shape}. "
            f"Expected (None, {MODEL_INPUT_SHAPE[0]}, {MODEL_INPUT_SHAPE[1]}, "
            f"{MODEL_INPUT_SHAPE[2]})."
        )

    if tuple(model.output_shape) != (None, EXPECTED_CLASS_COUNT):
        raise PredictionError(
            f"Unexpected model output shape: {model.output_shape}. "
            f"Expected (None, {EXPECTED_CLASS_COUNT})."
        )

    return model


@lru_cache(maxsize=1)
def load_class_names(class_names_path: Path = CLASS_NAMES_PATH) -> list[str]:
    """Load class names without reordering or modifying the JSON file."""
    class_names_path = Path(class_names_path)
    if not class_names_path.is_file():
        raise PredictionError(f"Class-name file was not found: {class_names_path}")

    try:
        with class_names_path.open("r", encoding="utf-8") as file:
            class_names = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        raise PredictionError(f"The class-name JSON could not be loaded: {exc}") from exc

    if not isinstance(class_names, list) or not all(
        isinstance(name, str) and name.strip() for name in class_names
    ):
        raise PredictionError("Class-name JSON must contain a non-empty list of strings.")

    if len(class_names) != EXPECTED_CLASS_COUNT:
        raise PredictionError(
            f"Found {len(class_names)} class names; expected {EXPECTED_CLASS_COUNT}. "
            "The model and class-name JSON must remain aligned."
        )

    return class_names


def predict_from_array(image_array: np.ndarray) -> list[dict[str, float | str]]:
    """Run a model prediction for a validated 224 x 224 RGB float32 array."""
    if not isinstance(image_array, np.ndarray):
        raise PredictionError("The image array must be a NumPy array.")

    expected_shape = (1, *MODEL_INPUT_SHAPE)
    if image_array.shape != expected_shape:
        raise PredictionError(
            f"Unexpected image batch shape: {image_array.shape}. "
            f"Expected {expected_shape}."
        )

    if not np.issubdtype(image_array.dtype, np.floating):
        raise PredictionError("The image array must use a floating-point dtype.")

    model = load_model()
    class_names = load_class_names()

    try:
        probabilities = model.predict(image_array, verbose=0)
    except Exception as exc:
        raise PredictionError(f"Prediction failed: {exc}") from exc

    probabilities = np.asarray(probabilities)
    if probabilities.shape != (1, EXPECTED_CLASS_COUNT):
        raise PredictionError(
            f"Unexpected prediction shape: {probabilities.shape}. "
            f"Expected (1, {EXPECTED_CLASS_COUNT})."
        )

    flat_probabilities = probabilities[0]
    top_indices = np.argsort(flat_probabilities)[::-1][:3]

    return [
        {
            "class_name": class_names[index],
            "probability": float(flat_probabilities[index]),
        }
        for index in top_indices
    ]
