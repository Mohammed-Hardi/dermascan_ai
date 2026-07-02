# Repository Guidelines

## Project Structure & Module Organization

DermaScan AI is a Streamlit + FastAPI + PyTorch project.

- `frontend/`: Streamlit UI, API client, styling, and visual assets in `frontend/assets/`.
- `backend/`: FastAPI app, routes, schemas, image-quality checks, inference, reports, and temporary scan storage.
- `ml/`: dataset preparation, model definitions, training, evaluation, Grad-CAM, export, and config files in `ml/configs/`.
- `tests/`: pytest suite for API behavior, image quality, datasets, metrics, ML pipeline, and model delivery.
- `docs/`: dataset card, model card, API docs, ethics notes, and implementation status.
- `run_app.bat` / `run_app.ps1`: one-command local app runner.

Generated data, raw images, `.venv`, and model outputs are ignored by Git.

## Build, Test, and Development Commands

Run the full app locally:

```powershell
.\run_app.bat
```

Run backend only:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Run frontend only:

```powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Train the current three-class baseline:

```powershell
.\.venv\Scripts\python.exe -m ml.src.train --config ml/configs/custom_cnn_three_class.yaml
```

## Coding Style & Naming Conventions

Use Python 3.11+, 4-space indentation, type hints where practical, and small focused modules. Prefer descriptive snake_case for functions/files and PascalCase for classes. Keep UI copy non-diagnostic and medically cautious. Do not hard-code local absolute paths in application code.

## Testing Guidelines

Tests use `pytest`. Name files `test_*.py` and keep tests close to the behavior being protected. Add or update tests when changing API routes, image validation, dataset logic, model loading, or report generation. Always run `pytest -q` before handing off changes.

## Commit & Pull Request Guidelines

This repository currently has no commit history, so no existing convention is enforced. Use short imperative commit messages, for example `Add three-class training config`. Pull requests should include a summary, test results, screenshots for UI changes, and notes about model/data limitations when ML behavior changes.

## Security & Configuration Tips

Use `.env.example` as the template for local settings. Keep raw datasets, trained checkpoints, `.env`, and generated reports out of Git unless explicitly approved. The app is educational only; preserve the required disclaimer on all result/report surfaces.
