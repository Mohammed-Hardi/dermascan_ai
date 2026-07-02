# Implementation Status

## Completed software

- Streamlit landing, camera/upload, consent, result, and report flow
- FastAPI health, prediction, report, and model-information endpoints
- Image validation and EXIF-stripping re-encoding
- Temporary bounded scan storage
- Safe explanation templates and disclaimers
- PDF reports
- SCIN download, validation, repair, provenance, and licence records
- Case-grouped 70/15/15 splits with duplicate and leakage audits
- Configurable augmentation and deterministic evaluation transforms
- Custom CNN, MobileNetV2, EfficientNet-B0, and ResNet50 factories
- Weighted training, early stopping, checkpoints, CSV, and TensorBoard logs
- Evaluation metrics, classification report, confusion matrix, ROC, and
  calibration output
- TorchScript export, CLI prediction, and Grad-CAM utility
- Placeholder/checkpoint inference selection with smoke-model safety gate
- Three-class SCIN training split and CPU-trained custom CNN baseline for
  eczema, tinea, and psoriasis
- Dockerfiles, Compose configuration, tests, API docs, dataset card, model card,
  and ethics documentation

## Research work not complete

- No full production-candidate training run has been completed.
- The current three-class checkpoint is a weak CPU baseline and is not valid
  clinical performance evidence.
- No external or Ghana-specific test set has been evaluated.
- No dermatologist-led error review or clinical validation has occurred.
- The current deployed baseline covers only eczema, tinea, and psoriasis.

The application can run the three-class checkpoint for academic demonstration,
but it must remain clearly non-diagnostic.
