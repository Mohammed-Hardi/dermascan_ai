# DermaScan AI Model Card

## Current status

A three-class custom CNN baseline has been trained on the local SCIN-derived
dataset for eczema, tinea, and psoriasis. This is an academic demonstration
checkpoint, not a clinically validated model.

The checkpoint is stored at
`ml/outputs/models/dermascan-three-class-custom-cnn.pt`. It completed 12 CPU
training epochs. The best checkpoint was selected by validation loss.

Test-set performance is weak and must not be presented as medical reliability:

- Accuracy: 0.3913
- Macro F1-score: 0.2905
- Weighted F1-score: 0.4375
- Macro ROC-AUC: 0.4783
- Expected calibration error: 0.0572

The older EfficientNet smoke-test checkpoint still exists only to prove that the
pipeline executes; it must not be reported as research performance.

## Intended model

- Current baseline architecture: custom CNN
- Candidate architecture for stronger future training: EfficientNet-B0
- Comparison models: MobileNetV2, ResNet50, custom CNN
- Current baseline classes: eczema, tinea, psoriasis
- Input: 160 x 160 RGB image for the custom CNN baseline
- Training approach: supervised image classification with class-weighted loss
- Loss: class-weighted cross entropy
- Optimizer: AdamW
- Scheduler: cosine annealing
- Selection criterion: validation loss with early stopping

## Intended use

The eventual model is intended for academic research, education, and early
screening support. It may present visually similar condition categories and
uncertainty to encourage appropriate professional follow-up.

## Not intended use

- Clinical diagnosis
- Medication or treatment prescription
- Emergency triage
- Autonomous medical decisions
- Replacement for a qualified healthcare professional

## Dataset

The current SCIN subset and grouped splits are documented in
`docs/DATASET_CARD.md`. The three-class baseline uses 1,544 total images:

- Eczema: 1,028 total, 725 train
- Tinea: 292 total, 204 train
- Psoriasis: 224 total, 158 train

The model must not be presented as Ghana-ready without additional evaluation on
representative Ghanaian skin tones and phone-camera images.

## Required evaluation before release

- Accuracy
- Macro precision, recall, and F1-score
- Weighted F1-score
- Per-class precision, sensitivity, specificity, and F1-score
- Top-three accuracy
- Confusion matrix
- One-vs-rest ROC-AUC where each class has positive and negative examples
- Expected calibration error
- Skin-tone subgroup analysis
- Error review by qualified domain experts

## Current limitations

- No production-candidate transfer-learning run has been completed.
- The current custom CNN baseline has poor test performance.
- No clinically validated external test set is available.
- The source dataset was collected in the United States rather than Ghana.
- Class imbalance remains substantial.
- Image-only classification cannot incorporate the full clinical context.
