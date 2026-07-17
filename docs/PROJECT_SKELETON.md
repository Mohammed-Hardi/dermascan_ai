# DermaScan AI Project Skeleton

This document explains the basic structure of the DermaScan AI project and how the frontend, backend, and machine learning model work together.

## 1. Project Overview

DermaScan AI is an educational skin screening system. A user uploads or captures a skin image, the backend prepares it automatically, and the system returns a possible class prediction for three supported conditions:

- Acne
- Scabies
- Psoriasis

The project is built with:

- Streamlit for the frontend user interface.
- FastAPI for the backend API.
- PyTorch for model training and prediction.
- A transfer-learned EfficientNet-B0 checkpoint for the three-class skin condition model.

## 2. Main Folder Structure

```text
DermaScan AI/
├── frontend/              # Streamlit user interface
├── backend/               # FastAPI backend application
├── ml/                    # Dataset preparation, training, evaluation, model code
├── results/               # Saved model metrics and evaluation outputs
├── docs/                  # Project documentation
├── tests/                 # Pytest test files
├── run_app.bat            # One-command Windows app runner
├── run_app.ps1            # PowerShell app runner
└── requirements.txt       # Python dependencies
```

## 3. Frontend Implementation

The frontend is implemented in `frontend/app.py`. It is the main Streamlit application that the user sees in the browser.

The frontend contains these main pages:

- Landing page: introduces DermaScan AI and the supported conditions.
- Scan page: allows the user to upload or capture an image and start analysis.
- Result page: shows the predicted condition, confidence score, probability breakdown, AI explanation, and safety disclaimer.
- About page: explains the project purpose and technology stack.

The design and colors are controlled in `frontend/styles.py`. This file defines the blue medical theme, background images, cards, buttons, and responsive layout.

The frontend communicates with the backend through `frontend/api.py`. This file sends uploaded images to the FastAPI backend and receives prediction results. For Streamlit Cloud deployment, it can also run a local fallback using the trained checkpoint if the backend server is not running separately.

## 4. Backend Implementation

The backend starts from `backend/app/main.py`. It creates the FastAPI application, enables CORS, and connects the API routes.

Important backend route files include:

- `backend/app/routes/predict.py`: receives an uploaded image and returns a prediction.
- `backend/app/routes/model_info.py`: returns model name, version, classes, and status.
- `backend/app/routes/report.py`: generates a PDF report for a scan result.

The prediction route works in this order:

1. The frontend sends the uploaded or captured image to `/api/v1/predict`.
2. The backend reads the uploaded file.
3. `image_quality.py` checks that the image is valid and acceptable.
4. The backend center-crops the accepted image automatically.
5. `inference.py` sends the prepared image to the trained model.
6. `explanation.py` creates simple AI consultant text for the predicted class.
7. The result is saved temporarily in `storage/scans.py`.
8. The backend sends the final response back to the frontend.

## 5. Model Implementation

The model code is stored in the `ml/` folder.

Key files include:

- `ml/src/models.py`: defines and creates the neural network model.
- `ml/src/train.py`: trains the model using the selected YAML config.
- `ml/src/datasets.py`: loads image paths and labels for training.
- `ml/src/augmentations.py`: applies image resizing and transformations.
- `ml/src/evaluate_basic.py`: evaluates the trained model without optional plotting dependencies.
- `ml/configs/accuracy_target_acne_scabies_psoriasis.yaml`: active three-class training configuration.

The trained checkpoint used by the app is:

```text
ml/outputs/models/dermascan-acne-scabies-psoriasis-efficientnet-b0.pt
```

The backend center-crops the skin image, and the model pipeline resizes it to the configured input size, converts it into a tensor, and predicts probabilities for acne, scabies, and psoriasis.

## 6. Model Prediction Flow

The full prediction flow is:

```text
User image
  ↓
Streamlit upload/camera input
  ↓
Frontend sends the original image to backend
  ↓
Backend validates image quality
  ↓
Backend automatically prepares and center-crops the image
  ↓
PyTorch model predicts class probabilities
  ↓
Backend builds explanation and disclaimer
  ↓
Frontend displays result to user
```

## 7. Model Performance

The saved evaluation report contains the main three performance metrics, but they are not displayed on the public website:

- Accuracy
- Validation accuracy
- Weighted F1-score

These values are read from:

```text
results/model_metrics.json
```

These values remain available for the project report and offline model evaluation.

## 8. Safety and Disclaimer

DermaScan AI is not a medical diagnosis system. It is designed as an educational and decision-support prototype. Every result page includes a disclaimer telling users to consult a qualified healthcare professional or dermatologist for proper diagnosis and treatment.

The system also warns users to seek urgent care if a rash is rapidly spreading, painful, bleeding, infected, or associated with fever.

## 9. How the Whole System Connects

In simple terms:

- The frontend handles user interaction and visual design.
- The backend handles API logic, validation, prediction, and reports.
- The ML folder handles training, evaluation, and model creation.
- The trained model checkpoint connects the ML work to the running app.

This separation makes the project easier to explain, test, and defend because each part has a clear responsibility.
