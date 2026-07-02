# DermaScan AI

DermaScan AI is a Ghana-focused, AI-assisted skin screening and education
project. It accepts a skin image, checks image quality, returns possible visual
categories, explains the result in non-diagnostic language, and generates a PDF
report.

## Safety status

The current application uses deterministic placeholder scores to exercise the
complete product flow. It is not a trained medical model and must not be used
for real screening decisions.

The scan result is not a diagnosis. For accurate diagnosis and treatment
recommendations, consult a qualified healthcare professional or dermatologist.

## Stack

- Streamlit frontend
- React Native mobile client in `mobile/`
- FastAPI backend
- PyTorch and timm model foundation
- Pillow and OpenCV image processing
- ReportLab PDF generation
- pytest test suite

## Setup

Python dependencies are installed in `.venv`. To recreate the environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the full app

Open one PowerShell terminal in the repository root:

```powershell
.\run_app.bat
```

This starts the FastAPI backend on `http://127.0.0.1:8000`, waits for it to be
ready, then starts the Streamlit frontend on `http://127.0.0.1:8501`. If the
three-class checkpoint exists, the runner uses it automatically; otherwise it
falls back to placeholder mode.

## Run the backend

Open a PowerShell terminal in the repository root:

```powershell
.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload --port 8000
```

API documentation is available at `http://127.0.0.1:8000/docs`.

## Run the Streamlit frontend

Open a second PowerShell terminal in the repository root:

```powershell
.venv\Scripts\Activate.ps1
streamlit run frontend/app.py --server.port 8501
```

Then open `http://127.0.0.1:8501`.

## Run the React Native mobile client

The mobile client is an Expo app in `mobile/`. Start the FastAPI backend first,
then run:

```powershell
cd mobile
npm install
npm run start
```

For a physical phone, set `expo.extra.apiBaseUrl` in `mobile/app.json` to your
computer LAN address, for example `http://192.168.1.20:8000`.

## Run tests

```powershell
.venv\Scripts\Activate.ps1
pytest -q
```

## Current API

- `GET /health`
- `POST /api/predict`
- `GET /api/report/{scan_id}`
- `GET /api/model-info`

## Dataset

The selected SCIN subset is documented in `docs/DATASET_CARD.md`. Raw images
are intentionally excluded from Git. Split by `case_id`, not individual image,
to prevent leakage between training, validation, and test sets.

Prepare or regenerate the validated dataset index and grouped splits with:

```powershell
python -m ml.src.datasets
```

Create the focused three-class split for eczema, tinea, and psoriasis:

```powershell
python -m ml.src.prepare_three_class_dataset
```

## Train a model

Run the EfficientNet-B0 transfer-learning configuration with:

```powershell
python -m ml.src.train --config ml/configs/efficientnet_b0.yaml
```

Run the current three-class CPU baseline with:

```powershell
python -m ml.src.train --config ml/configs/custom_cnn_three_class.yaml
```

Evaluate the trained three-class checkpoint with:

```powershell
python -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-three-class-custom-cnn.pt `
  --config ml/configs/custom_cnn_three_class.yaml `
  --split test
```

Available configurations are `efficientnet_b0.yaml`, `mobilenet_v2.yaml`,
`resnet50.yaml`, and `custom_cnn.yaml`. Training uses class-weighted cross
entropy, AdamW, cosine learning-rate scheduling, early stopping, CSV logs, and
TensorBoard logs.

Use a bounded pipeline check on CPU before a full training run:

```powershell
python -m ml.src.train --config ml/configs/efficientnet_b0.yaml --smoke-test
```

## Evaluate a model

Evaluate a completed checkpoint on the held-out test split with:

```powershell
python -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-efficientnet-b0.pt `
  --config ml/configs/efficientnet_b0.yaml `
  --split test
```

Evaluation writes JSON metrics, a CSV classification report, a confusion
matrix, and one-vs-rest ROC curves. Smoke checkpoints are written with
`smoke-` artifact prefixes and must not be reported as research performance.

## Export and inspect a checkpoint

```powershell
python -m ml.src.export_model `
  --model ml/outputs/models/dermascan-efficientnet-b0.pt `
  --format torchscript

python -m ml.src.predict `
  --model ml/outputs/models/dermascan-efficientnet-b0.pt `
  --image path/to/image.jpg
```

## Inference modes

The backend defaults to `DERMASCAN_INFERENCE_MODE=placeholder`. A completed
training checkpoint can be enabled with:

```text
DERMASCAN_INFERENCE_MODE=checkpoint
DERMASCAN_MODEL_PATH=ml/outputs/models/dermascan-efficientnet-b0.pt
DERMASCAN_ALLOW_SMOKE_MODEL=false
```

Smoke checkpoints are rejected unless `DERMASCAN_ALLOW_SMOKE_MODEL=true` is
set explicitly for development verification.

## Docker

Start the placeholder-mode application with:

```powershell
docker compose up --build
```

The Streamlit app is served at `http://127.0.0.1:8501` and the API at
`http://127.0.0.1:8000`.

## Next model milestone

The placeholder inference service will be replaced only after dataset review,
case-grouped splitting, model training, evaluation, and model-card updates are
complete. Scabies requires additional licensed training data before the model
can be considered a serious research baseline.
