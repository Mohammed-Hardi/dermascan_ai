# Model Accuracy Improvement Work

This note records the active acne, scabies, and psoriasis model and its honest held-out evaluation.

## Active Checkpoint

```text
ml/outputs/models/dermascan-acne-scabies-psoriasis-efficientnet-b0.pt
```

The model is EfficientNet-B0. It was initialized from the previous three-class checkpoint, its feature extractor was frozen, and its classification head was fine-tuned for acne, scabies, and psoriasis at `160 x 160` input resolution.

## Dataset

| Class | Real unique total | Real train | Validation | Test | Generated train samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Acne | 1,000 | 689 | 156 | 155 | 1,000 |
| Scabies | 917 | 649 | 121 | 147 | 1,000 |
| Psoriasis | 1,000 | 703 | 150 | 147 | 1,000 |

Training-only augmentation balances the manifest to 1,000 samples per class.
Validation and test images remain original. Source case or patient identifiers
are grouped into one partition, and there is zero exact SHA-256 overlap among
partitions.

## Results

- Test accuracy: **85.30%** (`383/449` correct)
- Validation accuracy: **86.42%**
- Weighted F1-score: **85.24%**
- Macro F1-score: **85.14%**
- Expected calibration error: **3.35%**

| Class | Precision | Recall | F1 | Test support |
| --- | ---: | ---: | ---: | ---: |
| Acne | 88.34% | 92.90% | 90.57% | 155 |
| Scabies | 86.52% | 82.99% | 84.72% | 147 |
| Psoriasis | 80.69% | 79.59% | 80.14% | 147 |

All three class F1-scores exceed 80% on this internal held-out test. The largest
remaining confusion is between scabies and psoriasis: 19 scabies images were
classified as psoriasis, and 17 psoriasis images were classified as scabies.
This model remains an academic screening prototype, not a clinically validated
diagnostic system.

## Reproduce

```powershell
python -m ml.src.prepare_balanced_three_class_dataset --images-per-class 1000 --seed 42
python -m ml.src.train --config ml/configs/accuracy_target_acne_scabies_psoriasis.yaml
python -m ml.src.evaluate_basic `
  --model ml/outputs/models/dermascan-acne-scabies-psoriasis-efficientnet-b0.pt `
  --config ml/configs/accuracy_target_acne_scabies_psoriasis.yaml `
  --split test
```

External evaluation on independently collected Ghanaian clinical images and
dermatologist-led error review are required before making clinical reliability
claims.
