# DermaScan AI Model Card

## Current Model

The active checkpoint is `ml/outputs/models/dermascan-acne-scabies-psoriasis-efficientnet-b0.pt`. It is a transfer-learned EfficientNet-B0 classifier with three outputs: acne, scabies, and psoriasis. The model consumes `160 x 160` RGB images.

It is an academic final-year project checkpoint, not a clinically validated diagnostic model.

## Training and Evaluation

The final run warm-started the existing EfficientNet-B0 checkpoint and trained
its 3,843-parameter classifier head for eight CPU epochs. Training used 1,000
samples per class, balanced with deterministic augmentation. Augmentation was
restricted to training.

Evaluation used 449 untouched real images. Splits are grouped by source case or
patient, with zero case overlap and zero exact SHA-256 overlap among training,
validation, and test data. The checkpoint was selected by lowest validation
loss at epoch 3; the test split was evaluated after model selection.

- Test accuracy: 85.30%
- Validation accuracy: 86.42%
- Weighted F1-score: 85.24%
- Macro F1-score: 85.14%
- Expected calibration error (10 bins): 3.35%
- Checkpoint size: 15.6 MB
- Local CPU latency: approximately 110 ms per image (20-run single-image test)

| Class | Precision | Recall | F1 | Test images |
| --- | ---: | ---: | ---: | ---: |
| Acne | 88.34% | 92.90% | 90.57% | 155 |
| Scabies | 86.52% | 82.99% | 84.72% | 147 |
| Psoriasis | 80.69% | 79.59% | 80.14% | 147 |

## Intended Use

- Academic research and demonstration
- Educational skin-screening support
- Showing possible visual categories with uncertainty and safety guidance

## Prohibited Use

- Clinical diagnosis or treatment prescription
- Emergency triage
- Autonomous medical decisions
- Replacement for a qualified healthcare professional

## Dataset

The curated pool contains 2,917 unique real images: 1,000 acne, 917 scabies,
and 1,000 psoriasis. It represents 940 acne cases, 639 scabies cases, and 881
psoriasis cases based on available source identifiers. The balanced training
manifest has 1,000 samples per class, but augmented copies are not new patients
or new clinical cases.

## Limitations

- Source data are not representative of the Ghanaian population.
- Labels and photographic protocols vary across SCIN, DermNet, the Mendeley
  scabies benchmark, and SkinDisNet.
- Some DermNet filenames provide class labels but not patient identifiers; each
  such file is conservatively treated as a separate case.
- No external clinical validation or dermatologist-led error review has been completed.
- Image-only classification cannot use symptoms, examination findings, or laboratory confirmation.

The interface must continue to state that every result is educational and not a diagnosis.
