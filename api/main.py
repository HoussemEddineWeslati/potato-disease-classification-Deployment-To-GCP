"""FastAPI service that classifies potato leaf images.

Run with the local model:          uvicorn api.main:app
Run against TensorFlow Serving:    PREDICTOR=tf-serving uvicorn api.main:app
"""

from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile

from .inference import InvalidImageError, KerasPredictor, Predictor, TFServingPredictor, preprocess, to_result

MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def create_app(predictor: Predictor) -> FastAPI:
    app = FastAPI(title="Potato disease classifier", version="2.0.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(file: UploadFile = File(...)) -> dict:
        if file.content_type and not file.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="upload an image file")
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 5 MB")
        try:
            batch = preprocess(data)
        except InvalidImageError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return to_result(predictor.predict(batch)[0])

    return app


def _default_predictor() -> Predictor:
    return TFServingPredictor() if os.environ.get("PREDICTOR") == "tf-serving" else KerasPredictor()


app = create_app(_default_predictor())
