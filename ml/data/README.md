# Dataset Workspace

The curated SCIN images are under `raw/<class>/`. Source metadata, the exact
download manifest, licence, and downloader are under `source/scin/`.

## Current layout

```text
raw/
  acne/
  eczema/
  tinea/
  scabies/
  psoriasis/
  other/
source/scin/
  LICENSE
  SOURCE_README.md
  scin_cases.csv
  scin_labels.csv
  download_manifest.csv
  download_scin.ps1
```

To resume or verify downloads on Windows:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ml/data/source/scin/download_scin.ps1 `
  -WorkerCount 96 -MaxRetries 4
```

The downloader skips existing non-empty files. The official SCIN release has
one documented missing image, so the raw folder can be complete even though
the original manifest contains one unavailable URL.

Do not split individual images at random. Use `case_id` from
`download_manifest.csv` to keep every image from one case in the same split.
See `docs/DATASET_CARD.md` for provenance, mapping, licence, and limitations.

## Prepare the dataset

Run validation, exact and perceptual duplicate detection, and deterministic
case-grouped splitting from the repository root:

```powershell
python -m ml.src.datasets
```

This writes the validated index and audit CSV files to `processed/`, then
creates `train.csv`, `val.csv`, `test.csv`, and `summary.json` in `splits/`.
The generated data files are ignored by Git and can be recreated from the
source manifest.
