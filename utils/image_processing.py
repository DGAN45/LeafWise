from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from config import ALLOWED_IMAGE_EXTENSIONS, MODEL_INPUT_SIZE


class ImageValidationError(ValueError):
    """Raised when an uploaded image is missing or invalid."""


def validate_image_upload(filename: str, file_size: int) -> None:
    """Validate the image filename extension and enforce a safe upload limit."""
    if not filename:
        raise ImageValidationError("Please select an image file.")

    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        raise ImageValidationError("Unsupported format. Upload a JPG, JPEG, or PNG image.")

    if file_size <= 0:
        raise ImageValidationError("The uploaded image is empty.")

    if file_size > 10 * 1024 * 1024:
        raise ImageValidationError("The image exceeds the 10 MB upload limit.")


def load_and_prepare_image(
    uploaded_file: BinaryIO,
    filename: str,
    file_size: int | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Validate and convert an uploaded image for model inference."""
    if file_size is None:
        try:
            file_size = uploaded_file.seek(0, 2)
            uploaded_file.seek(0)
        except (AttributeError, OSError) as exc:
            raise ImageValidationError("The uploaded image could not be read.") from exc

    validate_image_upload(filename, file_size)

    try:
        uploaded_file.seek(0)
        payload = uploaded_file.read()
        with Image.open(BytesIO(payload)) as source:
            source.verify()
        with Image.open(BytesIO(payload)) as source:
            image = source.convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageValidationError(
            "The file is not a readable JPG, JPEG, or PNG image."
        ) from exc

    if image.width < 32 or image.height < 32:
        raise ImageValidationError(
            "The image is too small. Please upload an image at least 32 x 32 pixels."
        )

    prepared_image = image.resize(MODEL_INPUT_SIZE, Image.Resampling.LANCZOS)
    image_array = np.asarray(prepared_image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    if image_array.shape != (1, 224, 224, 3):
        raise ImageValidationError("The image could not be converted to the required format.")

    return image.copy(), image_array
