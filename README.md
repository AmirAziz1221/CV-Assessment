# Visual Defect Detection – CV Assessment

Classifies product images as **normal** / **defective** and serves the model through a FastAPI service.
Built phase by phase; every number in `docs/` is produced by a script in this repo.

| Phase | Status | Where to look |
|---|---|---|
| 1 Dataset analysis | done | `docs/dataset_analysis.md` |
| 2 Data preparation | done | `docs/data_prep.md`, `docs/split_report.md` |
| 3 Modeling | next | |
| 4 Evaluation | | |
| 5 Export + API | | |
| 6 Docker / CI | | |
| 7 Final README, video script, interview Q&A | | |

> **Dataset note.** No client dataset was supplied, so development uses the public
> *Magnetic Tile Surface Defects* dataset (1,344 grayscale images: 952 normal, 392 defective).
> To use your own data, place images in `data/raw/normal/` and `data/raw/defective/` and skip `make data`.
> If the filenames contain `<defecttype>__name.jpg` the defect type is used for stratification and error analysis;
> otherwise it falls back to the class name.

## Quick start (Phases 1–2)

```bash
git clone https://github.com/AmirAziz1221/CV-Assessment.git && cd CV-Assessment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
# CPU-only machine? install the small CPU build of torch first:
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
make setup          # pip install -r requirements.txt
make prep           # = data + analyze + splits + resolution + augment-check   (a few minutes)
make test           # 17 unit tests
```
No `make`? Run the underlying commands (they are one-liners, see the `Makefile`):
`python scripts/prepare_data.py`, `python scripts/analyze_dataset.py`, `python scripts/make_splits.py`,
`python scripts/resolution_study.py`, `python scripts/augmentation_check.py`, `python -m pytest -q`.

## Repository layout
```
configs/default.yaml        all data settings (seed, resolution, augmentation)
data/                       raw images + split manifest (images are git-ignored; recreated by make data)
docs/                       dataset_analysis.md, data_prep.md, split_report.md, figures/
scripts/                    prepare_data, analyze_dataset, make_splits, resolution_study, augmentation_check
src/                        dupes.py (duplicate groups), splits.py, data.py (transforms, Dataset, imbalance helpers)
tests/                      pytest suite
```

## Key decisions so far (details and numbers in `docs/`)
1. **Split by tile, not image** – 68% of images are re-crops of the same tile; a random split leaks a near-copy
   for 60% of test images, the group split for 0.
2. **5 grouped CV folds + locked test (14%)** – ~674 independent tiles is too few for one noisy split.
3. **320 px, plain stretch** – defects are tiny; chosen from a pixel-size study (to be confirmed by a Phase 3 ablation).
4. **Defect-safe augmentation** – verified on pixel masks (no draw loses >10% of a defect); crops/shifts rejected.
