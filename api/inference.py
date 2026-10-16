"""Image preprocessing and the predictors behind the API.

The trained model contains its own Resizing and Rescaling(1/255) layers, so it
expects raw RGB pixels in 0-255. Preprocessing therefore only decodes, converts
to RGB and resizes to the training size so images can be batched; it must not
divide by 255 a second time.
"""

from __future__ import annotations

import io
import os
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

CLASS_NAMES = ("Early Blight", "Late Blight", "Healthy")
IMAGE_SIZE = 256


class InvalidImageError(ValueError):
    """The uploaded bytes are not a readable image."""


def preprocess(data: bytes) -> np.ndarray:
    """Decode an uploaded image into a (1, 256, 256, 3) float32 batch of raw pixels."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("the file is not a readable image") from error
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.asarray(image, dtype=np.float32)[np.newaxis, ...]


def to_result(probabilities: np.ndarray) -> dict:
    """Turn one row of class probabilities into the API response."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError(f"expected {len(CLASS_NAMES)} probabilities, got {probabilities.shape[0]}")
    best = int(np.argmax(probabilities))
    return {
        "class": CLASS_NAMES[best],
        "confidence": round(float(probabilities[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probabilities)},
    }


class Predictor(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray: ...


class KerasPredictor:
    """Loads the SavedModel once, on first use."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.environ.get("MODEL_PATH", "models/2")
        self._model = None

    def predict(self, batch: np.ndarray) -> np.ndarray:
        if self._model is None:
            import tensorflow as tf  # imported lazily so the API starts fast and tests do not need it

            self._model = tf.keras.models.load_model(self.model_path)
        return self._model.predict(batch, verbose=0)


class TFServingPredictor:
    """Sends the batch to a TensorFlow Serving REST endpoint."""

    def __init__(self, url: str | None = None, timeout: float = 10.0) -> None:
        self.url = url or os.environ.get("TF_SERVING_URL", "http://localhost:8501/v1/models/potatoes_model:predict")
        self.timeout = timeout

    def predict(self, batch: np.ndarray) -> np.ndarray:
        import requests

        response = requests.post(self.url, json={"instances": batch.tolist()}, timeout=self.timeout)
        response.raise_for_status()
        return np.asarray(response.json()["predictions"])
