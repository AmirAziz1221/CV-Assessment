# Phase 1 – Dataset analysis

> **Data provenance.** No client dataset was supplied, so this analysis uses the public
> **Magnetic Tile Surface Defects** dataset (Huang et al., "Saliency of magnetic tile surface defects",
> GitHub `abin24/Magnetic-tile-defect-datasets`). It was re-organised into `data/raw/normal` and
> `data/raw/defective` (the "MT_Free" folder = normal; Blowhole/Break/Crack/Fray/Uneven = defective;
> the defect type is kept as a filename prefix for error analysis). **All numbers below come from running
> `python scripts/analyze_dataset.py`; nothing is estimated.** Re-run it on the real data and every figure updates.

## 1. Class balance
| class | images | share |
|---|---|---|
| normal | 952 | 70.8% |
| defective | 392 | 29.2% |

Imbalance ratio **2.43 : 1** (normal : defective). Moderate – not extreme, so weighted loss / sampling is
enough; anomaly detection is not required. Defective images by type: blowhole 115, uneven 103, break 85,
crack 57, **fray 32** (the rarest type – expect weaker recall there).
Figure: `figures/class_counts.png`.

## 2. Image properties
- 1,344 files, **all JPEG, all single-channel grayscale ("L")**, **0 corrupt**.
  Pretrained backbones expect 3 channels → convert by replicating the channel.
- **Sizes are highly variable:** width 103–632 px, height 222–403 px, aspect ratio 0.36–2.68;
  **1,225 distinct sizes** among 1,344 images. Images are tile crops, not fixed-size camera frames.
  → A plain resize to a square distorts shape (tall tiles get squashed). This is a Phase 2 decision (see §7).
- Figure: `figures/image_properties.png`.

## 3. Shortcut check (can a trivial feature predict the label?)
AUC of one feature at a time for "defective" (0.5 = no information):

| feature | AUC |
|---|---|
| width | 0.495 |
| height | 0.429 |
| aspect ratio | 0.503 |
| area | 0.475 |
| brightness | 0.514 |
| contrast | 0.505 |
| sharpness | 0.534 |

No feature is strongly predictive. The largest deviation is `height` (0.43, defective images are slightly
shorter on average: 311 vs 323 px mean). Weak, but worth remembering: if preprocessing keeps image height
information (e.g. padding instead of resizing), a model could pick up a little of it. Per-class means of
brightness/contrast are nearly identical (≈110 / 30.5).

## 3b. Duplicates – the most important finding
- **Exact duplicate files (MD5): 0.**
- **dHash (64-bit) screening over-flags:** 5,525 pairs within 4 bits. I checked and this hash is unreliable on
  these smooth, low-texture images, so it is only a screen.
- **Thumbnail correlation (32×32 grayscale, Pearson) is the authoritative check.** Viewing pairs shows the
  **same physical tile cropped at slightly different sizes** (e.g. 478×363 vs 489×371), with identical
  scratches and shading. Images connected by correlation > 0.98 form a *duplicate group*:
  - **908 of 1,344 images (68%) sit in a group of 2+ images**; 238 such groups; largest group = 12 images.
  - The 1,344 images collapse into **674 distinct physical groups** (groups containing normal images: 456; containing defective images: 221; 3 groups contain both).
  - Sensitivity (`stats.json`): largest group is 12 for every threshold 0.98–0.995; at 0.95 unrelated tiles
    start chaining (largest group 86), so 0.95 is too loose. I use **0.98** deliberately on the cautious side:
    missing a duplicate = leakage; over-grouping only costs a little split flexibility.
  - **3 groups mix normal and defective crops of the same tile.** I cannot tell from the images whether this is
    label noise or a defect lying outside the "normal" crop. Either way these images must stay together.
- Figures: `figures/duplicate_group_examples.png`; group table: `figures/dup_groups.csv`.

**Leakage risk: HIGH if split naively.** A random image-level split would place crops of the same tile in
train and test, inflating test metrics. The effective sample size is ~674 independent tiles, not 1,344 images.

## 4. What defects look like (from 392 pixel masks)
Defect area as a fraction of the image (median):

| type | median | range |
|---|---|---|
| blowhole | 0.14% | 0.03–0.67% |
| crack | 0.31% | 0.07–2.3% |
| break | 1.3% | 0.09–23% |
| fray | 15% | 0.1–44% |
| uneven | 25% | ~0–48% |

Overall median 0.49%; the smallest 10% of defects cover < 0.086% of the image.
- **Tiny, subtle defects (blowhole, crack):** at 224×224 a median blowhole is ≈ 72 px² (≈ 8×8 px); the
  smallest decile is ≈ 6×6 px. Resolution matters → test ≥ 224 and larger inputs in Phase 3.
- **Large, low-contrast defects (uneven, fray):** broad brightness/texture changes; need global context, and
  tolerate downscaling.
- **Location:** defects occur across the whole frame including edges (`figures/defect_size_location.png`)
  → **never centre-crop**. Density is highest bottom-centre, lowest at mid-left/right.
- Visual notes (`figures/samples_defective.png`): tile surfaces carry vertical brushing lines, strong
  lighting gradients, and some tiles have handwritten numbers / marker marks – a possible spurious cue
  to check with Grad-CAM later.

## 5. Candidate risks for the model
1. Duplicate groups → group-aware splitting is mandatory.
2. Tiny defects → resolution-sensitive; small-defect recall is the likely weak point.
3. Variable aspect ratios → choose resizing strategy deliberately.
4. Small effective data (~674 groups) → report confidence intervals; single split is noisy.
5. Rare "fray" class (32 images) → very few test examples per type.

## 6. Files produced
`scripts/analyze_dataset.py`, `docs/figures/{class_counts,image_properties,samples_normal,samples_defective,duplicate_group_examples,defect_size_location}.png`,
`docs/figures/{stats.json,image_properties.csv,dup_groups.csv}`.

## 7. Implications carried into Phase 2 (proposed – to confirm)
- Split by **duplicate group** (`StratifiedGroupKFold`, seed fixed), not by image.
- Because ~674 groups is small: **5-fold grouped CV on train+val** plus a locked group-disjoint test set,
  report mean ± std / bootstrap CI.
- Resizing: compare plain resize vs. aspect-preserving letterbox at 224 and ~320 px before locking.
