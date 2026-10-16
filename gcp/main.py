"""Google Cloud Function serving the Keras model stored in a Cloud Storage bucket.

Deploy:  gcloud functions deploy predict --runtime python312 --trigger-http --memory 1024MB \
           --set-env-vars BUCKET_NAME=<your-bucket>,MODEL_BLOB=models/potatoes.h5
"""

import io
import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]
IMAGE_SIZE = 256
_model = None


def _load_model():
    global _model
    if _model is None:
        import tensorflow as tf
        from google.cloud import storage

        local_path = "/tmp/model.h5"
        bucket = storage.Client().bucket(os.environ["BUCKET_NAME"])
        bucket.blob(os.environ.get("MODEL_BLOB", "models/potatoes.h5")).download_to_filename(local_path)
        _model = tf.keras.models.load_model(local_path)
    return _model


def predict(request):
    upload = request.files.get("file")
    if upload is None:
        return {"error": "send the image in a 'file' form field"}, 400
    try:
        image = Image.open(io.BytesIO(upload.read())).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except OSError:
        return {"error": "the file is not a readable image"}, 400

    # The model rescales pixels itself, so raw 0-255 values are passed in.
    batch = np.asarray(image, dtype=np.float32)[np.newaxis, ...]
    probabilities = _load_model().predict(batch, verbose=0)[0]
    best = int(np.argmax(probabilities))
    return {"class": CLASS_NAMES[best], "confidence": round(float(probabilities[best]), 4)}
