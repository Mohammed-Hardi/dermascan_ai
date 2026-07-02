# DermaScan AI Code Walkthrough for Defense

## 1. Project Overview

DermaScan AI is an academic skin screening support system. It allows a user to upload or capture a skin image, checks whether the image is clear enough, sends it to an AI model, and returns possible skin condition categories with confidence scores. The system is not a medical diagnosis tool. Every result includes a disclaimer telling the user to consult a qualified healthcare professional or dermatologist.

The current model focuses on three classes:

| Class | Training Images Used |
| --- | ---: |
| Acne | 900 |
| Eczema | 900 |
| Psoriasis | 900 |

The dataset source is the SCIN dataset. The project uses a balanced augmented training set. This means the system starts from real SCIN images, then creates extra transformed versions of the training images to balance all classes to 900 images each. The validation and test sets remain separate from the training data.

## 2. Main Folder Structure

The project is divided into clear folders:

```text
backend/       FastAPI backend for prediction, reports, and model info
frontend/      Streamlit user interface
ml/            Dataset preparation, model training, evaluation, and export
docs/          Project documentation and report notes
tests/         Automated tests
run_app.bat    One-command app runner for Windows
run_app.ps1    PowerShell script that starts backend and frontend
```

The frontend and backend are separated because the frontend handles user interaction, while the backend handles validation, AI inference, and report generation.

## 3. How The Application Runs

The easiest way to run the project is:

```powershell
.\run_app.bat
```

This calls `run_app.ps1`. The PowerShell script does three main things:

1. It checks whether the trained checkpoint exists:

```text
ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt
```

2. If the checkpoint exists, it sets the backend to checkpoint inference mode.

3. It starts:

```text
FastAPI backend: http://127.0.0.1:8000
Streamlit frontend: http://127.0.0.1:8501
```

This means the user only needs one command to run the full system locally.

## 4. Frontend Code Explanation

The main frontend file is:

```text
frontend/app.py
```

This file builds the Streamlit interface. Streamlit is used because it allows a Python-based web app without writing a separate JavaScript frontend.

The frontend has three main pages:

1. Home page
2. Scan/upload page
3. Result page

### Home Page

The home page introduces DermaScan AI and explains that the system is for educational screening support. It also shows the background image and a button to start the scan.

Important functions:

```python
render_home()
render_brand()
render_disclaimer()
```

`render_disclaimer()` is important because the project must never present the AI output as a diagnosis.

### Upload And Analyze Page

The upload page is created by:

```python
render_scan()
```

This page allows the user to either:

- Upload an image file
- Capture an image using the camera

The code uses:

```python
st.file_uploader()
st.camera_input()
```

The user must tick a consent checkbox before analysis. This makes the user acknowledge that the system is not a diagnosis tool.

Optional user context is also collected:

- Age range
- Sex
- Body location
- Symptom duration

These values are sent to the backend as metadata. They are not used to diagnose, but they can appear in the report.

### Result Page

The result page is created by:

```python
render_result()
```

It shows:

- Uploaded image
- Top predicted class
- Confidence score
- Top possible classes
- AI explanation
- PDF download button
- Disclaimer

The frontend does not run the AI model directly. It calls the backend through helper functions in:

```text
frontend/api.py
```

## 5. Frontend API Helper

The file:

```text
frontend/api.py
```

contains small functions that communicate with the backend.

Important functions:

```python
health_check()
get_model_info()
predict_image()
download_report()
```

`predict_image()` sends the selected image to:

```text
POST /api/predict
```

`download_report()` gets the PDF report from:

```text
GET /api/report/{scan_id}
```

This keeps the Streamlit code cleaner because API communication is separated from UI code.

## 6. Frontend Styling

The file:

```text
frontend/styles.py
```

contains custom CSS for the Streamlit app.

It controls:

- Page background
- Buttons
- Hero section
- Upload section image
- Result cards
- Disclaimer boxes

The images are stored in:

```text
frontend/assets/
```

The CSS embeds the images as base64 data. This helps the images load correctly when the app runs locally.

## 7. Backend Entry Point

The main backend file is:

```text
backend/app/main.py
```

This creates the FastAPI application.

It defines:

```python
app = FastAPI(...)
```

It also enables CORS using:

```python
CORSMiddleware
```

CORS allows the Streamlit frontend at port `8501` to communicate with the FastAPI backend at port `8000`.

The backend exposes:

```text
GET /health
POST /api/predict
GET /api/report/{scan_id}
GET /api/model-info
```

`/health` is a simple endpoint used to check if the backend is running.

## 8. Configuration Code

The file:

```text
backend/app/config.py
```

stores important settings.

Examples:

- Maximum upload size
- Minimum image dimension
- Blur threshold
- Brightness threshold
- Confidence threshold
- Model path
- Inference mode

The backend can run in two modes:

```text
placeholder
checkpoint
```

`placeholder` returns fake deterministic scores for development. `checkpoint` loads the trained PyTorch model.

For the real project demo, the app should use checkpoint mode with:

```text
dermascan-acne-eczema-psoriasis-custom-cnn.pt
```

## 9. Prediction Route

The most important backend route is:

```text
backend/app/routes/predict.py
```

This handles the image prediction request.

The flow is:

1. Receive uploaded image
2. Read the image bytes
3. Validate image quality
4. Run model inference
5. Generate safe explanation
6. Save temporary scan record
7. Return prediction response

The route is:

```python
@router.post("/predict", response_model=PredictionResponse)
```

If the image is not valid, the backend returns an error instead of making a prediction.

## 10. Image Quality Checking

The file:

```text
backend/app/services/image_quality.py
```

checks whether the uploaded image is suitable for analysis.

It checks:

- File is not empty
- File size is not too large
- File is JPG, PNG, or WEBP
- Image can be opened
- Image dimensions are large enough
- Image is not too dark
- Image is not too bright
- Image is not too blurry

Blur is measured with OpenCV using the Laplacian variance method. Brightness is measured from the grayscale version of the image.

The image is also converted to RGB and saved again as a clean JPEG. This removes EXIF metadata and reduces privacy risk.

## 11. Model Inference

The file:

```text
backend/app/services/inference.py
```

contains the prediction logic.

There are two prediction functions:

```python
predict_placeholder()
predict_checkpoint()
```

`predict_placeholder()` is only for development. It creates deterministic fake results from the image bytes.

`predict_checkpoint()` loads the trained model and runs real inference.

The real inference steps are:

1. Load the checkpoint model
2. Resize and normalize the image
3. Pass the image tensor through the neural network
4. Apply softmax to convert model outputs into probabilities
5. Select the top 3 predictions
6. Return class names and confidence scores

The model currently predicts among:

```text
acne, eczema, psoriasis
```

## 12. Model Loading

The file:

```text
backend/app/models/model_loader.py
```

loads the PyTorch checkpoint from disk.

It checks that the checkpoint contains important fields:

- `model_name`
- `model_state_dict`
- `class_names`
- `input_size`

Then it rebuilds the same model architecture and loads the saved weights.

The model is cached so the backend does not reload it from disk for every request.

## 13. Response Schemas

The file:

```text
backend/app/schemas.py
```

defines the structure of data sent between backend and frontend.

Important schemas:

```python
PredictionItem
Explanation
PredictionResponse
QualityResult
ModelInfo
HealthResponse
```

Using schemas helps FastAPI validate the response and automatically generate API documentation.

## 14. Explanation And Safety Text

The file:

```text
backend/app/services/explanation.py
```

creates simple explanations for each prediction.

The explanation is written carefully. It does not say:

```text
You have acne.
```

Instead it says the image appears visually similar to a category and advises professional follow-up.

The required disclaimer is:

```text
The scan result is not a diagnosis. For accurate diagnosis and treatment recommendations, consult a qualified healthcare professional or dermatologist.
```

This disclaimer appears in the frontend result and PDF report.

## 15. Temporary Storage

The file:

```text
backend/app/storage/scans.py
```

stores scan results temporarily in memory.

It stores:

- Scan ID
- Image bytes
- Prediction result
- Explanation
- Metadata
- Model version

The storage is process-local and limited to a maximum number of records. It is not a permanent database. This is safer for a student project because sensitive medical images are not permanently stored by default.

## 16. PDF Report Generation

The file:

```text
backend/app/services/pdf_report.py
```

uses ReportLab to generate a PDF report.

The report includes:

- Scan ID
- Date
- Model version
- Submitted image thumbnail
- Top screening result
- Possible classes and confidence scores
- Explanation
- Disclaimer

The route:

```text
backend/app/routes/report.py
```

returns the generated PDF to the frontend.

## 17. Dataset Preparation

Dataset preparation code is in:

```text
ml/src/datasets.py
ml/src/prepare_three_class_dataset.py
ml/src/prepare_balanced_three_class_dataset.py
```

`datasets.py` contains the general dataset logic. It can:

- Read image metadata
- Validate image files
- Compute hashes
- Detect duplicates
- Create train, validation, and test splits
- Load images for PyTorch training

`prepare_three_class_dataset.py` filters the dataset to the selected classes.

`prepare_balanced_three_class_dataset.py` creates the current balanced training dataset:

```text
acne: 900 training images
eczema: 900 training images
psoriasis: 900 training images
```

The balanced set is created by deterministic augmentation from the SCIN train split only. This prevents validation and test images from leaking into training.

## 18. Data Augmentation

Data augmentation means creating modified versions of existing images to help the model learn better.

The project uses transformations such as:

- Horizontal flip
- Small rotation
- Cropping
- Brightness adjustment
- Contrast adjustment
- Color adjustment

Augmentation is useful because acne and psoriasis had fewer original images than eczema. However, it must be explained honestly: augmented images are not the same as collecting new real patient images.

## 19. Model Architecture

The file:

```text
ml/src/models.py
```

defines the available neural network models.

The current baseline uses:

```text
custom_cnn
```

The custom CNN contains:

- Convolution layers
- Batch normalization
- ReLU activation
- Max pooling
- Dropout
- Linear classifier

The project also includes support for:

```text
EfficientNet-B0
MobileNetV2
ResNet50
```

These are stronger transfer-learning models, but they are better trained on a GPU.

## 20. Training Code

The main training file is:

```text
ml/src/train.py
```

Training uses:

- PyTorch
- Cross entropy loss
- Class weights
- AdamW optimizer
- Cosine learning rate scheduler
- Early stopping
- TensorBoard logs
- Checkpoint saving

The training config used for the current model is:

```text
ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml
```

This config defines:

- Model name
- Number of classes
- Class names
- Image size
- Batch size
- Number of epochs
- Learning rate
- Dataset CSV paths
- Output folders

The training command is:

```powershell
.\.venv\Scripts\python.exe -m ml.src.train --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml
```

The saved model checkpoint is:

```text
ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt
```

## 21. Evaluation Code

The evaluation file is:

```text
ml/src/evaluate.py
```

It loads the trained checkpoint and tests it on the test split.

It calculates:

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Top-3 accuracy
- ROC-AUC
- Confusion matrix
- Expected calibration error

The evaluation command is:

```powershell
.\.venv\Scripts\python.exe -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt `
  --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml `
  --split test
```

The outputs are saved under:

```text
ml/outputs/reports/acne_eczema_psoriasis/
ml/outputs/plots/acne_eczema_psoriasis/
```

## 22. Grad-CAM Code

The file:

```text
ml/src/gradcam.py
```

contains a Grad-CAM utility. Grad-CAM is used to show which parts of an image influenced the model decision.

In the current project, the Grad-CAM utility exists, but the heatmap is not fully integrated into the Streamlit result page yet. This can be explained as future work.

## 23. Testing Code

The tests are stored in:

```text
tests/
```

Important test files include:

- `test_api.py`
- `test_image_quality.py`
- `test_datasets.py`
- `test_metrics.py`
- `test_ml_pipeline.py`
- `test_model_delivery.py`

The tests check that important parts of the system work correctly, such as:

- API response format
- Image validation
- Dataset splitting
- Metric calculation
- Model export
- Prediction pipeline

Run all tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 24. Full User Flow From Start To End

This is the complete system flow:

1. The user opens the Streamlit app.
2. The user uploads or captures a skin image.
3. The frontend previews the image.
4. The user confirms the tool is not a diagnosis.
5. The frontend sends the image to FastAPI.
6. FastAPI checks image quality.
7. If the image is valid, FastAPI loads the trained model.
8. The model predicts acne, eczema, or psoriasis.
9. The backend returns the top prediction and confidence scores.
10. The frontend displays the result.
11. The user can download a PDF report.
12. The result includes the medical disclaimer.

## 25. What To Say During Defense

You can explain the project like this:

> My project is an AI-assisted skin screening support system. The user uploads a skin image through a Streamlit interface. The image is sent to a FastAPI backend, where it is checked for quality before being passed to a trained PyTorch CNN model. The model currently classifies images into acne, eczema, and psoriasis. The system then returns possible categories with confidence scores and generates a PDF report. It is designed for educational support only and clearly states that the result is not a medical diagnosis.

For the dataset, say:

> I used the SCIN dataset and created a balanced three-class training set for acne, eczema, and psoriasis. Each class has 900 training images. Since acne and psoriasis had fewer original images, I used deterministic augmentation on the training split only. Validation and testing are kept separate to avoid data leakage.

For safety, say:

> The system avoids giving treatment advice or diagnosis. It uses non-diagnostic wording and always advises the user to consult a qualified healthcare professional or dermatologist.

## 26. Current Limitations

The project is functional, but it still has limitations:

- It is not clinically validated.
- It should not be used as a real medical diagnosis system.
- Acne and psoriasis were balanced using augmentation, not 900 unique patient images.
- The current baseline model is a custom CNN trained on CPU.
- A stronger model such as EfficientNet should be trained on a GPU in future work.
- Grad-CAM heatmap display is prepared but not fully integrated into the UI.
- More Ghana-specific validation data would improve the research value.

## 27. Future Improvements

Possible improvements include:

- Train EfficientNet-B0 on a GPU.
- Add Grad-CAM heatmap display to the result page.
- Add a persistent database for scan history if privacy approval is available.
- Deploy the backend separately on Render, Railway, or AWS.
- Deploy the frontend on Streamlit Community Cloud.
- Collect or evaluate with Ghana-specific dermatology images.
- Add dermatologist review of model errors.

## 28. Key Commands To Remember

Run the app:

```powershell
.\run_app.bat
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Prepare balanced dataset:

```powershell
.\.venv\Scripts\python.exe -m ml.src.prepare_balanced_three_class_dataset --images-per-class 900
```

Train model:

```powershell
.\.venv\Scripts\python.exe -m ml.src.train --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml
```

Evaluate model:

```powershell
.\.venv\Scripts\python.exe -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-acne-eczema-psoriasis-custom-cnn.pt `
  --config ml/configs/custom_cnn_acne_eczema_psoriasis_balanced.yaml `
  --split test
```

## 29. Final Summary

DermaScan AI combines a Streamlit frontend, FastAPI backend, PyTorch model, image quality checking, safe explanation generation, and PDF reporting. The project demonstrates how AI can support early skin screening education while still respecting medical safety boundaries. The strongest point of the system is that it separates user interface, backend logic, model training, and documentation clearly, making it easier to explain, test, and improve.
