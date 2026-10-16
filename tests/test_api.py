import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings, strategies as st
from PIL import Image

from api.inference import CLASS_NAMES, IMAGE_SIZE, InvalidImageError, preprocess, to_result
from api.main import create_app


class FakePredictor:
    """Returns fixed probabilities and records the batches it received."""

    def __init__(self, probabilities=(0.1, 0.7, 0.2)):
        self.probabilities = np.array([probabilities])
        self.batches = []

    def predict(self, batch):
        self.batches.append(batch)
        return self.probabilities


def image_bytes(size=(300, 200), mode="RGB", fmt="PNG"):
    buffer = io.BytesIO()
    Image.new(mode, size, color=120 if mode == "L" else (10, 200, 30)).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def predictor():
    return FakePredictor()


@pytest.fixture
def client(predictor):
    return TestClient(create_app(predictor))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_the_most_likely_class_with_all_probabilities(client):
    response = client.post("/predict", files={"file": ("leaf.png", image_bytes(), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["class"] == "Late Blight"
    assert body["confidence"] == 0.7
    assert set(body["probabilities"]) == set(CLASS_NAMES)


@pytest.mark.parametrize("mode,fmt", [("RGB", "PNG"), ("RGBA", "PNG"), ("L", "JPEG"), ("RGB", "JPEG")])
def test_any_image_is_resized_to_the_training_shape_with_raw_pixels(mode, fmt):
    batch = preprocess(image_bytes(size=(640, 480), mode=mode, fmt=fmt))
    assert batch.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)
    assert batch.dtype == np.float32
    assert batch.max() > 1.0  # not divided by 255: the model rescales itself


def test_a_file_that_is_not_an_image_is_rejected_with_400(client, predictor):
    response = client.post("/predict", files={"file": ("notes.png", b"not an image", "image/png")})
    assert response.status_code == 400
    assert predictor.batches == []


def test_a_non_image_content_type_is_rejected_with_415(client):
    assert client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 415


def test_an_oversized_upload_is_rejected_with_413(client):
    big = b"\x00" * (5 * 1024 * 1024 + 1)
    assert client.post("/predict", files={"file": ("big.png", big, "image/png")}).status_code == 413


def test_a_wrong_number_of_probabilities_is_an_error():
    with pytest.raises(ValueError):
        to_result(np.array([0.5, 0.5]))


@given(st.binary(max_size=2048))
@settings(max_examples=200)
def test_random_bytes_are_either_a_valid_image_or_a_clean_error(data):
    try:
        batch = preprocess(data)
    except InvalidImageError:
        return
    assert batch.shape == (1, IMAGE_SIZE, IMAGE_SIZE, 3)


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=3, max_size=3))
def test_the_reported_class_is_always_the_highest_probability(values):
    result = to_result(np.array(values))
    assert result["probabilities"][result["class"]] == max(result["probabilities"].values())
    assert 0.0 <= result["confidence"] <= 1.0
