# DermaScan AI Dataset Card

## Current dataset

The current development dataset is a curated subset of the Skin Condition
Image Network (SCIN) dataset. SCIN contains consented image donations from
United States internet users, dermatologist condition labels, and skin-tone
metadata. It is intended for health education and research.

- Source: https://github.com/google-research-datasets/scin
- DOI: https://doi.org/10.5281/zenodo.10819503
- Official bucket: `gs://dx-scin-public-data`
- Download date: 2026-06-22
- Local image count: 2,239 unique, readable PNG files
- Local case count: 1,039 contributions
- Local size: approximately 2.65 GB

## Class distribution

| DermaScan class | Images | Cases |
| --- | ---: | ---: |
| Acne | 118 | 58 |
| Eczema | 1,028 | 471 |
| Tinea / fungal infection | 292 | 134 |
| Scabies | 47 | 22 |
| Psoriasis | 224 | 105 |
| Other | 530 | 249 |

The counts represent unique files, not manifest rows. Three manifest rows
refer to an image already selected elsewhere in the same class. One official
SCIN image is unavailable, as documented by SCIN issue #1; that image would
have belonged to the tinea class.

## Label mapping

Cases were selected when the first (highest-weighted) entry in SCIN's
`weighted_skin_condition_label` matched the following mapping:

| DermaScan class | SCIN primary labels |
| --- | --- |
| Acne | `Acne` |
| Eczema | `Eczema` |
| Tinea / fungal infection | `Tinea`, `Tinea Versicolor`, `Fungal dermatosis` |
| Scabies | `Scabies` |
| Psoriasis | `Psoriasis` |
| Other | First 250 labelled, non-target cases in metadata order |

All available images belonging to each selected case were downloaded. Dataset
splits must therefore be grouped by `case_id`, not by image, to prevent images
from the same contribution appearing in multiple splits.

## Quality checks

- All 2,239 files can be decoded by the Windows image library.
- No zero-byte files are present.
- An interrupted initial transfer left 21 truncated acne images. They were
  detected during full pixel decoding and replaced from the official bucket.
- No exact SHA-256 duplicate files were found in the selected subset.
- Combined 64-bit dHash and pHash checks found no credible perceptual duplicate
  copies at distances 4 and 6 respectively. dHash-only matches were rejected
  after visual audit showed collisions between unrelated skin photographs.

## Reproducible splits

Splits were generated with seed `42`. Every image from a `case_id` remains in a
single partition. Any future same-class perceptual duplicate cases will also be
grouped into one partition.

| Split | Images | Image share | Cases |
| --- | ---: | ---: | ---: |
| Train | 1,579 | 70.52% | 728 |
| Validation | 331 | 14.78% | 156 |
| Test | 329 | 14.69% | 155 |

Generated files live under `ml/data/processed/` and `ml/data/splits/`. They are
excluded from Git because they are reproducible from the source manifest using
`python -m ml.src.datasets`.

## Three-class training subset

For the current retraining request, a focused three-class dataset was created
from the validated SCIN splits. It keeps the original case-grouped train,
validation, and test partitions, then filters to eczema, tinea, and psoriasis.

| Class | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Eczema | 725 | 152 | 151 | 1,028 |
| Tinea / fungal infection | 204 | 42 | 46 | 292 |
| Psoriasis | 158 | 33 | 33 | 224 |
| **Total** | **1,087** | **227** | **230** | **1,544** |

The generated three-class split files live under
`ml/data/splits_three_class/` and can be regenerated with:

```powershell
python -m ml.src.prepare_three_class_dataset
```

## Licence and obligations

SCIN is distributed under the SCIN Data Use Public License. A copy is stored at
`ml/data/source/scin/LICENSE`. The licence permits reproduction, sharing, and
adaptation subject to its terms. Important requirements include attribution,
retaining a licence and warranty notice, identifying modifications, avoiding
additional downstream restrictions, and never attempting to re-identify or
re-link participants.

This summary is not a replacement for reading the full licence.

## Limitations

- SCIN contributors were located in the United States, not Ghana.
- Class imbalance is severe, especially for scabies and acne.
- Forty-seven scabies images are not enough for a reliable final classifier.
- The `other` class is heterogeneous and requires manual label review.
- Dermatologist labels are differential/weighted labels, not necessarily
  biopsy-confirmed diagnoses.
- Skin-tone representation must be measured before claiming suitability for
  Ghanaian users.
- This dataset and any resulting model are for research, education, and
  screening support only, not clinical diagnosis.

## Work required before training

1. Manually audit class mappings and image relevance.
2. Add a legally usable source for scabies and, preferably, acne.
3. Manually review a representative sample from every split.
4. Report skin-tone distributions for every split.
5. Keep the raw images out of Git and preserve source attribution.
