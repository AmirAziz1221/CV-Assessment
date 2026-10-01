# Phase 2 – Data preparation

All numbers are produced by the scripts named in each section (`make prep` re-creates them).
Dataset: public Magnetic Tile Surface Defects (see `dataset_analysis.md` for provenance).

## 1. Splitting  (`scripts/make_splits.py`, `src/splits.py`, report: `split_report.md`)

**Rule: split by tile, not by image.** Phase 1 showed 68% of images are re-crops of a tile that also
appears elsewhere. We group images with thumbnail correlation > 0.98 (`src/dupes.py`) and put every group
into exactly one split. Stratification uses the **defect type** (6 values), so even the rarest type
(fray, 32 images) appears in every fold and in the test set.

| role | images | tiles | defective |
|---|---|---|---|
| 5 CV folds (train+val pool) | 229–232 each | 113–117 | 66–69 (≈29%) |
| locked test | 191 (14.2%) | 96 | 55 (28.8%) |

* **70/15/15:** train on 4 folds (~69%), validate on 1 (~17%), test ~14%. The test set is never touched
  while tuning (seed 42, manifest `data/splits.csv`).
* **Why k-fold:** with ~674 independent tiles one validation split is noisy. 5 grouped folds give mean ± std.
* **Leakage evidence** (test images having a near-identical twin on the train/val side):

| method | test images with a twin in train/val |
|---|---|
| **group-aware (ours)** | **0 of 191** |
| naive random image split | **114 of 191 (60%)** |

* **Uncertainty:** with 55 defective test images, a 95% Wilson interval on recall is about ±8 points at
  recall 0.90 — and this is optimistic, because crops of one tile are not independent. Phase 4 reports
  group-bootstrap intervals; differences of a few points between models will not be significant.

## 2. Resolution and resize mode  (`scripts/resolution_study.py`)

Defects are tiny (median blowhole ≈ 0.14% of the image), so before training I measured how many pixels a
defect keeps after resizing (side of a square with equal area, using the 388 non-empty pixel masks):

| input | mode | median defect side | defects < 10 px | relative compute |
|---|---|---|---|---|
| 224 | letterbox | 11.5 px | 40.2% | 1.0× |
| 224 | stretch | 15.6 px | 24.5% | 1.0× |
| 320 | letterbox | 16.5 px | 19.3% | 2.0× |
| **320** | **stretch** | **22.3 px** | **11.6%** | **2.0×** |
| 384 | stretch | 26.8 px | 5.9% | 2.9× |

* Letterbox (keep aspect ratio) wastes ≈36% of the canvas on padding, so at equal cost **stretch gives
  defects ~35% more pixels**, and it also removes image size/shape as a weak shortcut (height AUC 0.43 in Phase 1).
* **Chosen default: stretch @ 320 px** (`configs/default.yaml`). 320 is the compromise: 224 leaves a quarter
  of defects under 10 px; 384+ nearly triples CPU cost for a single-core machine.
* **This is a geometric argument, not yet an accuracy result.** Stretch distorts narrow tiles (visible in
  `figures/augmentation_samples.png`). Phase 3 trains both modes (and 224 vs 320) and reports the difference.

## 3. Preprocessing  (`src/data.py`)
Grayscale JPEG → replicate to 3 channels → resize → scale to [0,1] → ImageNet mean/std normalisation
(the pretrained backbones expect this). Identical for train, val, test and the API.

## 4. Augmentation (train only)

| augmentation | setting | reason |
|---|---|---|
| horizontal + vertical flip | p = 0.5 each | lossless; brushing lines stay vertical. **Assumption: tile orientation can vary on the line — confirm with the client** |
| rotation | ±7°, `expand=True` | camera/placement jitter; canvas grows so nothing is cut off |
| brightness / contrast | ±20% | lighting varies strongly in the data |
| gaussian blur | σ ≤ 1.0, p = 0.3 | mild focus variation |
| gaussian noise | σ = 0.02, p = 0.3 | sensor noise |

**Deliberately not used:** random crops / shifts (can push a tiny edge defect out of frame, keeping the
label "defective"), cutout / random erasing (erases defects), hue/saturation (images are grayscale),
elastic / heavy warps (cracks and blowholes would change shape), strong blur (hides blowholes).

**Safety test** (`scripts/augmentation_check.py`): apply the geometric augmentations jointly to each image and
its defect mask, 50 draws × 333 training images that have a non-empty mask = 16,650 draws, and measure how much defect area survives:

| geometric pipeline | worst case retained | draws losing > 10% of defect | draws losing > 50% |
|---|---|---|---|
| my first design (flips + rotation + 3% shift) | 0% | 18.4% | 3.6% |
| random crop 70–100% area (considered) | 0% | 47.5% | 15.4% |
| **final (flips + rotation with expand)** | **95.2%** | **0%** | **0%** |

The first design looked reasonable by eye but the test showed it created wrong labels (worst for `break`
defects, which sit on tile edges: 35% of draws lost >10%). It was replaced before any training.
Samples: `figures/augmentation_samples.png`.

## 5. Class imbalance (2.43 : 1)
Two tools are implemented and unit-tested in `src/data.py`:
* `class_weights` – loss weights N / (2·n_c): ≈ 0.70 (normal) and ≈ 1.7 (defective).
* `balanced_sampler` – draws ~50/50 batches; because augmentation is random, repeated minority images are
  different variants (= oversampling with augmentation).

**Comparison against the unweighted baseline requires training, so it is done in Phase 3** (and judged with
recall / F1 / PR-AUC, never accuracy).

## 6. What to confirm with the client
1. Can tiles appear flipped / rotated on the line? (justifies vertical flip)
2. Is aspect-ratio distortion acceptable? (stretch vs letterbox, settled by the Phase 3 ablation)
3. The 3 groups with both normal and defective crops of the same tile: label noise or a defect outside the crop?
