from datetime import datetime, timezone
from pathlib import Path
import threading

import pandas as pd

from config import HISTORY_PATH


HISTORY_COLUMNS = [
    "timestamp",
    "plant_name",
    "disease_name",
    "predicted_class",
    "confidence",
]

_history_lock = threading.Lock()


def ensure_history_file(history_path: Path = HISTORY_PATH) -> Path:
    """Create an empty prediction-history CSV with the standard columns."""
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    if not history_path.is_file():
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(history_path, index=False)

    return history_path


def load_history(history_path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Load history safely and always return the documented column structure."""
    history_path = ensure_history_file(history_path)

    try:
        history = pd.read_csv(history_path)
    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        history = pd.DataFrame(columns=HISTORY_COLUMNS)

    for column in HISTORY_COLUMNS:
        if column not in history.columns:
            history[column] = pd.NA

    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(
            history["timestamp"],
            errors="coerce",
            utc=True,
        )

    if "confidence" in history.columns:
        history["confidence"] = pd.to_numeric(
            history["confidence"],
            errors="coerce",
        )

    return history.reindex(columns=HISTORY_COLUMNS)


def add_history_record(
    plant_name: str,
    disease_name: str,
    predicted_class: str,
    confidence: float,
    history_path: Path = HISTORY_PATH,
) -> pd.DataFrame:
    """Append one prediction record and return the newest-first history."""
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Confidence must be a probability between 0 and 1.")

    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "plant_name": str(plant_name).strip(),
                "disease_name": str(disease_name).strip(),
                "predicted_class": str(predicted_class).strip(),
                "confidence": confidence,
            }
        ]
    )

    with _history_lock:
        history_path = ensure_history_file(history_path)
        existing = load_history(history_path)
        combined = pd.concat([existing, record], ignore_index=True)
        combined.reindex(columns=HISTORY_COLUMNS).to_csv(history_path, index=False)

    return load_history(history_path).sort_values(
        "timestamp",
        ascending=False,
        na_position="last",
    )


def clear_history(history_path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Permanently clear prediction records from the local CSV."""
    history_path = Path(history_path)
    with _history_lock:
        pd.DataFrame(columns=HISTORY_COLUMNS).to_csv(history_path, index=False)
    return load_history(history_path)
