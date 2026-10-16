# Potato disease classification: from training to deployment

[![CI](https://github.com/HoussemEddineWeslati/potato-disease-classification-Deployment-To-GCP/actions/workflows/ci.yml/badge.svg)](https://github.com/HoussemEddineWeslati/potato-disease-classification-Deployment-To-GCP/actions/workflows/ci.yml)

A convolutional neural network that recognises **early blight**, **late blight** and **healthy**
potato leaves from a photo, served through a tested FastAPI service, TensorFlow Serving or a
Google Cloud Function, with TensorFlow Lite versions for mobile.

## What's inside

| Part | Description |
|---|---|
| `Training/` | Notebooks: data loading and augmentation, CNN training, evaluation, TF Lite conversion (post-training and quantisation-aware) |
| `models/` | Two versions of the trained model, in SavedModel format |
| `tf-lite-models/` | Quantised TF Lite models |
| `api/` | FastAPI service, with a pluggable predictor: local Keras model or TensorFlow Serving |
| `gcp/` | Google Cloud Function that loads the model from Cloud Storage |
| `tests/` | API and preprocessing tests, including property-based tests |

## The API

```text
GET  /health    → {"status": "ok"}
POST /predict   (multipart form, field "file")
     → {"class": "Late Blight", "confidence": 0.97,
        "probabilities": {"Early Blight": 0.02, "Late Blight": 0.97, "Healthy": 0.01}}
```

- Rejects files that are not images (415), unreadable images (400) and uploads over 5 MB (413).
- Converts any image to RGB and resizes it to the training size (256 × 256).
- Passes **raw 0–255 pixels**: the model contains its own `Rescaling(1/255)` layer, so normalising
  again in the API would silently degrade accuracy.
- The model is loaded once, on first use. The predictor is injected, so the API is tested without TensorFlow.

```bash
pip install -r api/requirements.txt
uvicorn api.main:app                                  # uses models/2
MODEL_PATH=models/1 uvicorn api.main:app              # another model version
PREDICTOR=tf-serving uvicorn api.main:app             # forward to TensorFlow Serving
```

### With TensorFlow Serving

```bash
docker run -p 8501:8501 -v "$PWD:/potato-disease" tensorflow/serving \
  --rest_api_port=8501 --model_config_file=/potato-disease/models.config
```

`models.config` serves every model version under the name `potatoes_model`.

## Deploying on Google Cloud

```bash
cd gcp
gcloud functions deploy predict --runtime python312 --trigger-http --memory 1024MB \
  --set-env-vars BUCKET_NAME=<your-bucket>,MODEL_BLOB=models/potatoes.h5
```

The function downloads the model from the bucket on the first request and keeps it in memory.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Besides endpoint and preprocessing tests, two **property-based tests** (Hypothesis) check that
random bytes always give either a correctly shaped image batch or a clean error, never a crash,
and that the reported class is always the one with the highest probability.

## Training data

[PlantVillage dataset](https://www.kaggle.com/arjuntejaswi/plant-village) on Kaggle, potato classes only.

## Credits

This project started from the [codebasics](https://www.youtube.com/@codebasics) deep learning
project. I trained the models, then rebuilt the serving side: a tested FastAPI service with input
validation and a pluggable predictor, a fix for double pixel normalisation in the cloud function,
current dependencies and continuous integration.

**Houssem Eddine Weslati** · [houssemweslati.com](https://houssemweslati.com)
