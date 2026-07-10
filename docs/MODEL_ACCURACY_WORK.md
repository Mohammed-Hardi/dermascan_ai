# Model Accuracy Improvement Work

This note records the current model accuracy status and the changes made to support an 80%+ accuracy attempt.

## Current Baseline

The newly retrained checkpoint is:

```text
ml/outputs/models/dermascan-acne-eczema-psoriasis-efficientnet-b0.pt
```

Saved test metrics in `results/model_metrics.json`:

- Accuracy: 87.9%
- Validation accuracy: 81.5%
- Weighted F1-score: 87.8%

The current class recalls from `results/classification_report.csv` are:

- Acne recall: 86.0%
- Eczema recall: 96.1%
- Psoriasis recall: 81.3%

This clears the project target of 80%+ test accuracy on the held-out test split.

## Dataset Status

The balanced training split contains:

```text
ml/data/balanced_three_class/train.csv
```

- Acne: 1000 training images
- Eczema: 1000 training images
- Psoriasis: 1000 training images

Validation and test splits are smaller:

- Acne: 150 images
- Eczema: 154 images
- Psoriasis: 150 images

## Changes Made

The project now supports torchvision transfer-learning models without requiring `timm`.

Updated file:

```text
ml/src/models.py
```

Added support for:

- `torchvision_efficientnet_b0`
- `torchvision_mobilenet_v2`
- `torchvision_resnet18`
- `torchvision_resnet50`

Also fixed stale three-class configs that still referenced `tinea`.

New target config:

```text
ml/configs/accuracy_target_efficientnet_b0.yaml
```

This config trains EfficientNet-B0 on acne, eczema, and psoriasis using the balanced dataset.

## Recommended Training Command

Use this command for the 80%+ accuracy attempt:

```powershell
.\.venv\Scripts\python.exe -m ml.src.train --config ml/configs/accuracy_target_efficientnet_b0.yaml
```

Then evaluate:

```powershell
.\.venv\Scripts\python.exe -m ml.src.evaluate `
  --model ml/outputs/models/dermascan-acne-eczema-psoriasis-efficientnet-b0.pt `
  --config ml/configs/accuracy_target_efficientnet_b0.yaml `
  --split test
```

If the new model reaches the target, replace the deployed checkpoint path and update `results/model_metrics.json`.

## Important Note

An 80%+ score cannot be honestly guaranteed by code changes alone. It depends on dataset quality, class separation, training time, and whether pretrained weights are available. The best technical path is transfer learning with EfficientNet-B0 or MobileNet, trained with enough epochs and evaluated on the held-out test split.
