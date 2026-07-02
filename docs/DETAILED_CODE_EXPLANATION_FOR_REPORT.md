# Detailed Code Explanation for Final Year Report

## 1. Purpose of This Document

This document explains the DermaScan AI code step by step in simple language. It is written to help with project defense and report writing. The aim is not to memorize every line, but to understand what each file does, why it exists, and how the system works from the user opening the app to the model returning a result.

DermaScan AI is an AI-assisted skin screening support system. It accepts a skin image, checks the image quality, runs a trained deep learning model, returns possible skin condition categories, and generates a PDF report. The system is educational and non-diagnostic.

Current classes:

| Class | Training Images |
| --- | ---: |
| Acne | 900 |
| Eczema | 900 |
| Psoriasis | 900 |

The dataset used is based on SCIN. A balanced training dataset was created using augmentation so that each class has 900 training images.

## 2. High-Level System Flow

The system works like this:

1. User opens the Streamlit web app.
2. User uploads or captures a skin image.
3. Frontend sends the image to the backend.
4. Backend checks image quality.
5. Backend loads the trained PyTorch model.
6. Model predicts acne, eczema, or psoriasis.
7. Backend returns class probabilities and explanation.
8. Frontend displays the result.
9. User can download a PDF report.

In simple terms:

```text
User -> Streamlit frontend -> FastAPI backend -> PyTorch model -> Result + PDF
```

## 3. Root Project Files

### `README.md`

This file explains the project overview, how to run the app, how to train the model, and how to test the system. It is the first file a new developer or supervisor should read.

### `requirements.txt`

This file lists the Python libraries needed by the project. Examples include:

- `streamlit` for the frontend
- `fastapi` for the backend
- `torch` and `torchvision` for deep learning
- `opencv-python-headless` for image quality checks
- `reportlab` for PDF report generation
- `pytest` for testing

When setting up the project, dependencies are installed using:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `.env.example`

This file shows example environment variables. It tells the backend which model to load and whether to run in placeholder mode or checkpoint mode.

Important values:

```text
DERMASCAN_INFERENCE_MODE=checkpoint
DERMASCAN_MODEL_PATH=ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt
```

This means the app should use the trained model checkpoint.

### `run_app.bat`

This is the easiest command for Windows users. It starts the full app.

Command:

```powershell
.\run_app.bat
```

The `.bat` file simply calls the PowerShell runner.

### `run_app.ps1`

This file starts both backend and frontend.

It does the following:

1. Finds the project folder.
2. Finds the Python virtual environment.
3. Checks if the trained model exists.
4. Sets model environment variables.
5. Starts FastAPI on port `8000`.
6. Waits until the backend is ready.
7. Starts Streamlit on port `8501`.

Important idea:

```text
run_app.ps1 makes the project run with one command.
```

## 4. Frontend Folder

The frontend is stored in:

```text
frontend/
```

The frontend is built with Streamlit.

## 5. `frontend/app.py`

This is the main Streamlit app file. It controls what the user sees.

### Imports

The file imports Streamlit and helper functions:

```python
import streamlit as st
from frontend.api import ApiError, download_report, get_model_info, health_check, predict_image
from frontend.styles import apply_styles
```

Meaning:

- `streamlit` displays the web interface.
- `predict_image()` sends image data to the backend.
- `download_report()` downloads the generated PDF report.
- `apply_styles()` applies custom CSS styling.

### Disclaimer

The app defines a disclaimer:

```python
DISCLAIMER = (
    "The scan result is not a diagnosis..."
)
```

This is important because the system must not claim to diagnose disease.

### `SelectedImage`

```python
@dataclass(slots=True)
class SelectedImage:
    data: bytes
    filename: str
    content_type: str
```

This class stores the uploaded or camera image in a clean structure. It keeps:

- The image bytes
- The filename
- The content type

### `initialize_state()`

This function prepares Streamlit session variables:

```python
page
result
scan_image
report_bytes
```

Streamlit reruns the script many times, so session state is used to remember what page the user is on and what result was returned.

### `navigate(page)`

This function changes the current page and reruns the app.

Example:

```python
navigate("scan")
```

This moves the user to the scan page.

### `render_brand()`

This displays the DermaScan AI logo/name area at the top of the app.

### `render_disclaimer()`

This displays the medical disclaimer in a styled box.

### `render_model_notice()`

This checks the backend model status by calling:

```python
get_model_info()
```

It tells the user whether the app is using:

- Placeholder mode
- Smoke test model
- Real checkpoint model
- Missing model

This improves transparency.

### `render_home()`

This builds the landing page. It shows:

- Project title
- Short explanation
- Background image
- Button to start scanning
- Four-step workflow
- Disclaimer

The button:

```python
st.button("GET INSTANT RESULT")
```

moves the user to the scan page.

### `_selected_image(uploaded_file, camera_file)`

This helper chooses either the camera image or uploaded image. It returns a `SelectedImage` object.

If the user has not selected any image, it returns `None`.

### `render_scan()`

This is the upload and analysis page.

It shows:

- Heading
- Upload tab
- Camera tab
- Image preview
- Optional metadata fields
- Consent checkbox
- Analyze button

Upload code:

```python
uploaded = st.file_uploader(...)
```

Camera code:

```python
captured = st.camera_input("Take a photo")
```

Consent checkbox:

```python
consent = st.checkbox("I understand this tool is not a diagnosis.")
```

The analyze button is disabled until:

1. An image is selected.
2. The user accepts the disclaimer.

When the user clicks analyze, the frontend calls:

```python
result = predict_image(...)
```

This sends the image and metadata to the backend.

### `render_result()`

This page displays the prediction.

It shows:

- Submitted image
- Top prediction
- Confidence
- Top possible categories
- Explanation
- PDF download button
- Scan another image button

It downloads the report using:

```python
download_report(result["scan_id"])
```

### `main()`

This is the entry point of the Streamlit app.

It:

1. Sets the page title and layout.
2. Applies CSS.
3. Initializes state.
4. Displays the correct page.

## 6. `frontend/api.py`

This file handles communication between frontend and backend.

### `API_URL`

```python
API_URL = os.getenv("DERMASCAN_API_URL", "http://127.0.0.1:8000")
```

This means the frontend will call the backend at port `8000` unless another URL is configured.

### `health_check()`

Checks whether the backend is running.

It calls:

```text
GET /health
```

### `get_model_info()`

Gets information about the model from:

```text
GET /api/model-info
```

This helps the frontend show whether the model is ready.

### `predict_image()`

This sends an image to:

```text
POST /api/predict
```

It sends:

- Image file
- Age range
- Sex
- Body location
- Symptom duration

If the backend returns an error, it raises `ApiError`.

### `download_report()`

This downloads the PDF report from:

```text
GET /api/report/{scan_id}
```

## 7. `frontend/styles.py`

This file contains the custom CSS for the app.

It controls:

- Background colors
- Buttons
- Hero image
- Upload section image
- Result cards
- Disclaimer design

It also converts local images into base64 so Streamlit can display them properly.

## 8. Backend Folder

The backend is stored in:

```text
backend/app/
```

It is built with FastAPI.

FastAPI was chosen because it is fast, simple, and automatically generates API documentation.

## 9. `backend/app/main.py`

This is the backend entry point.

It creates the FastAPI app:

```python
app = FastAPI(...)
```

It adds CORS middleware so the Streamlit frontend can call the backend.

It defines:

```python
@app.get("/health")
```

The health endpoint returns:

```json
{"status": "ok"}
```

It also includes routers:

```python
app.include_router(predict.router, prefix=settings.api_prefix)
app.include_router(report.router, prefix=settings.api_prefix)
app.include_router(model_info.router, prefix=settings.api_prefix)
```

This adds prediction, report, and model information routes.

## 10. `backend/app/config.py`

This file stores backend settings.

Examples:

- Maximum upload size
- Minimum image dimension
- Blur threshold
- Brightness limits
- Confidence threshold
- Model path
- Model name
- Allowed frontend origins

It uses `pydantic-settings`, which allows values to come from environment variables.

This is useful because the app can run differently in development and deployment.

## 11. `backend/app/schemas.py`

This file defines data models using Pydantic.

Important schemas:

### `PredictionItem`

Stores one prediction:

```python
class_name
confidence
```

### `Explanation`

Stores explanation text:

```python
summary
next_steps
warning
```

### `PredictionResponse`

This is the full response returned to the frontend.

It includes:

- Scan ID
- Top prediction
- Top 3 predictions
- Confidence level
- Risk level
- Explanation
- Disclaimer
- Model version
- Created date

### `QualityResult`

Stores image quality information:

- Is acceptable
- Reason if rejected
- Width
- Height
- Brightness
- Blur score

### `ModelInfo`

Returns model status to the frontend.

## 12. `backend/app/routes/predict.py`

This is the most important backend route.

Route:

```python
@router.post("/predict")
```

The function receives:

- Uploaded image
- Optional metadata

Steps:

1. Read image bytes.
2. Validate the image.
3. Reject the image if it is bad quality.
4. Run the model.
5. Build explanation text.
6. Store the scan temporarily.
7. Return prediction response.

This file connects image upload, quality checking, inference, explanation, and storage.

## 13. `backend/app/services/image_quality.py`

This file protects the model from poor input.

It checks:

- Empty file
- File too large
- Unsupported file type
- Invalid image
- Image too small
- Too dark
- Too bright
- Too blurry

Blur is calculated with:

```python
cv2.Laplacian(gray, cv2.CV_64F).var()
```

If the blur score is too low, the image is rejected.

The image is re-saved as a clean JPEG, which removes EXIF metadata and improves privacy.

## 14. `backend/app/services/inference.py`

This file performs prediction.

It contains:

```python
predict_placeholder()
predict_checkpoint()
predict()
```

### Placeholder Prediction

Placeholder mode is only for development. It creates fake but consistent results from the image bytes.

### Checkpoint Prediction

Checkpoint mode loads the trained PyTorch model.

Steps:

1. Select CPU or GPU.
2. Load model checkpoint.
3. Transform image to tensor.
4. Run model.
5. Apply softmax.
6. Get top 3 predictions.
7. Return prediction result.

The current deployed model should use checkpoint mode.

## 15. `backend/app/models/model_loader.py`

This file loads the model checkpoint.

It checks that the checkpoint has:

- Model name
- Model weights
- Class names
- Input size

It rebuilds the model architecture using:

```python
create_model()
```

Then it loads the saved weights.

The model is cached using `lru_cache`, so it is not loaded from disk every time.

## 16. `backend/app/services/explanation.py`

This file builds safe explanation text.

It avoids diagnosis language.

For example, instead of saying:

```text
You have eczema.
```

it says the image is visually similar to a category and recommends professional consultation.

This is important for medical ethics.

## 17. `backend/app/storage/scans.py`

This file stores scan records temporarily.

It uses an in-memory dictionary.

Each scan has:

- Scan ID
- Date
- Image bytes
- Prediction
- Explanation
- Metadata

This is not a permanent database. It is safer because medical images are not stored forever.

## 18. `backend/app/services/pdf_report.py`

This file generates the PDF report using ReportLab.

The PDF contains:

- Scan ID
- Date
- Model version
- Image thumbnail
- AI screening result
- Confidence table
- Explanation
- Disclaimer

This gives the user a document they can share with a health professional.

## 19. Machine Learning Folder

The machine learning code is stored in:

```text
ml/src/
```

This folder contains dataset preparation, augmentation, model architecture, training, evaluation, prediction, export, and Grad-CAM utilities.

## 20. `ml/src/datasets.py`

This file handles general dataset processing.

It defines:

```python
SkinConditionDataset
```

This class allows PyTorch to load images and labels from CSV files.

It also includes helper functions for:

- Image hashing
- Duplicate detection
- Dataset indexing
- Train/validation/test splitting

The project uses grouped splitting to reduce leakage. Images from the same case should not appear in both training and testing.

## 21. `ml/src/prepare_three_class_dataset.py`

This file filters the dataset to the selected classes.

Current classes:

```python
["acne", "eczema", "psoriasis"]
```

It reads the existing split CSV files and creates new three-class split files.

It remaps labels:

```text
acne -> 0
eczema -> 1
psoriasis -> 2
```

## 22. `ml/src/prepare_balanced_three_class_dataset.py`

This file creates the balanced training dataset used now.

It creates:

```text
900 acne training images
900 eczema training images
900 psoriasis training images
```

Why this was needed:

- Eczema had many images.
- Acne and psoriasis had fewer original images.
- To balance training, augmentation was used.

Important: augmentation is applied only to the training split. Validation and test data are not augmented in this script.

This helps avoid data leakage.

## 23. `ml/src/augmentations.py`

This file defines transformations used during training and evaluation.

Training transforms may include:

- Resize
- Random crop
- Horizontal flip
- Rotation
- Color jitter
- Blur
- Normalization

Evaluation transforms are more stable:

- Resize
- Normalize

The reason is that training needs variety, but evaluation should be consistent.

## 24. `ml/src/models.py`

This file defines model architectures.

The current baseline is:

```text
custom_cnn
```

The custom CNN uses:

- Convolution layers to extract image features
- Batch normalization to stabilize training
- ReLU activation to add non-linearity
- Max pooling to reduce image size
- Dropout to reduce overfitting
- Linear layer for classification

The project also supports:

```text
efficientnet_b0
mobilenet_v2
resnet50
```

These are transfer-learning models and can be used for stronger future training.

## 25. `ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml`

This file controls the current training setup.

Important values:

```yaml
model_name: custom_cnn
num_classes: 3
class_names: [acne, eczema, psoriasis]
input_size: 160
batch_size: 32
epochs: 6
```

It also points to the dataset CSV files:

```yaml
train_csv: ml/data/balanced_three_class/train.csv
val_csv: ml/data/balanced_three_class/val.csv
test_csv: ml/data/balanced_three_class/test.csv
```

## 26. `ml/src/train.py`

This file trains the AI model.

Main steps:

1. Load YAML config.
2. Set random seed.
3. Build train and validation datasets.
4. Create model.
5. Calculate class weights.
6. Define loss function.
7. Define optimizer.
8. Train for several epochs.
9. Validate after each epoch.
10. Save the best checkpoint.

Loss function:

```python
nn.CrossEntropyLoss()
```

Optimizer:

```python
AdamW
```

Scheduler:

```python
CosineAnnealingLR
```

The model is saved when validation loss improves.

The output checkpoint is:

```text
ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt
```

## 27. `ml/src/evaluate.py`

This file tests the trained model.

It loads:

- The checkpoint
- The test dataset
- The model architecture

Then it calculates:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- ROC curve
- Calibration error

This helps measure how well the model performs on unseen test images.

## 28. `ml/src/metrics.py`

This file contains metric calculations.

It calculates:

- Overall accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted F1
- Top-3 accuracy
- Per-class sensitivity
- Per-class specificity
- ROC-AUC
- Expected calibration error

These metrics are useful for the report because accuracy alone is not enough, especially when classes are imbalanced.

## 29. `ml/src/predict.py`

This file allows prediction from the command line using a saved checkpoint.

It is useful for testing one image without running the full web app.

## 30. `ml/src/export_model.py`

This file exports a trained PyTorch checkpoint to TorchScript.

TorchScript is useful when deploying models in environments that need a serialized model format.

## 31. `ml/src/gradcam.py`

This file contains Grad-CAM logic.

Grad-CAM helps show which part of the image influenced the model prediction.

In this project, the Grad-CAM utility exists, but the heatmap is not yet fully displayed in the Streamlit UI. This is future work.

## 32. Tests Folder

The tests are stored in:

```text
tests/
```

Important tests:

| Test File | Purpose |
| --- | --- |
| `test_api.py` | Tests API endpoints |
| `test_image_quality.py` | Tests image validation |
| `test_datasets.py` | Tests dataset logic |
| `test_metrics.py` | Tests metric calculations |
| `test_ml_pipeline.py` | Tests ML pipeline |
| `test_model_delivery.py` | Tests model export/reload |

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 33. Complete Prediction Journey

This is what happens when a user uploads an image:

1. User selects image in Streamlit.
2. Streamlit stores the image bytes.
3. User clicks Analyze.
4. `frontend/api.py` sends the image to FastAPI.
5. FastAPI receives the file in `predict.py`.
6. `image_quality.py` validates the image.
7. `inference.py` loads the model and predicts.
8. `explanation.py` builds safe explanation.
9. `scans.py` stores the scan temporarily.
10. FastAPI returns JSON result.
11. Streamlit displays the result.
12. User downloads PDF if needed.

## 34. Commands To Defend

Run app:

```powershell
.\run_app.bat
```

Prepare balanced dataset:

```powershell
.\.venv\Scripts\python.exe -m ml.src.prepare_balanced_three_class_dataset --images-per-class 900
```

Train:

```powershell
.\.venv\Scripts\python.exe -m ml.src.train --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml
```

Evaluate:

```powershell
.\.venv\Scripts\python.exe -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt `
  --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml `
  --split test
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 35. Simple Defense Explanation

You can say:

> This project is a web-based AI skin screening support system. The frontend is built with Streamlit, and the backend is built with FastAPI. The user uploads a skin image, the backend checks if the image is clear, then a PyTorch CNN model predicts whether the image is most similar to acne, eczema, or psoriasis. The system returns confidence scores and generates a PDF report. It is not a diagnosis system, so every result includes a clear medical disclaimer.

For the model:

> I used the SCIN dataset and created a balanced training set with 900 images per class. For classes with fewer original images, I used deterministic augmentation on training images only. The model used is a custom CNN trained with PyTorch.

For safety:

> The system avoids giving treatment or diagnosis. It only provides screening support and advises the user to consult a qualified healthcare professional.

## 36. Limitations

The project has limitations:

- It is not clinically validated.
- It should not replace a dermatologist.
- Some classes were balanced using augmentation, not new unique patient images.
- The current model is a CPU-trained custom CNN baseline.
- More real Ghana-specific images would improve the project.
- A stronger transfer-learning model should be trained on a GPU in future work.

## 37. Final Summary

DermaScan AI combines frontend design, backend API development, image processing, machine learning, PDF reporting, and medical safety principles. The code is organized so each part has a clear responsibility. This makes the project easier to explain, maintain, test, and improve.
