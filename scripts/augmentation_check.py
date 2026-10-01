"""Phase 2, step 3: show the augmentations AND prove they do not erase defects.

1. docs/figures/augmentation_samples.png : what the model actually sees (original + 5 augmented variants)
2. Safety test: apply the geometric augmentations jointly to each image and its pixel MASK, many times,
   and measure how much of the defect area survives. Compared with a random-crop alternative, which we
   rejected - the numbers show why.

Run:  python scripts/augmentation_check.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import tv_tensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import IMAGENET_MEAN, IMAGENET_STD, build_transform, geometric_aug  # noqa: E402

REPEATS = 50  # augmentation draws per defective image


def denorm(t):
    m, s = torch.tensor(IMAGENET_MEAN)[:, None, None], torch.tensor(IMAGENET_STD)[:, None, None]
    return (t * s + m).clamp(0, 1).permute(1, 2, 0).numpy()


def sample_figure(cfg, man, out):
    torch.manual_seed(0)
    tf = build_transform(cfg["data"], train=True)
    picks = [man[man.defect_type == t].iloc[3] for t in ["blowhole", "crack", "break", "fray", "uneven", "free"]]
    fig, ax = plt.subplots(len(picks), 6, figsize=(14, 2.3 * len(picks)))
    for r, row in enumerate(picks):
        img = Image.open(row.path).convert("RGB")
        ax[r][0].imshow(img); ax[r][0].set_title(f"original ({row.defect_type})", fontsize=8)
        for c in range(1, 6):
            ax[r][c].imshow(denorm(tf(img))); ax[r][c].set_title(f"aug {c}", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)


def safety_test(cfg, man):
    torch.manual_seed(0); np.random.seed(0)
    geo = geometric_aug(cfg["data"])
    rows = []
    for _, r in man[man.label == "defective"].iterrows():
        mp = Path("data/masks") / (Path(r.path).stem + ".png")
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L")) > 127
        if m.sum() == 0:
            continue
        img = tv_tensors.Image(torch.from_numpy(np.asarray(Image.open(r.path).convert("RGB"))).permute(2, 0, 1))
        mask = tv_tensors.Mask(torch.from_numpy(m.astype(np.uint8)))
        h, w = m.shape
        rng = np.random.default_rng(0)
        for _ in range(REPEATS):
            _, mk = geo(img, mask)                                  # same random params applied to image and mask
            kept_ours = min(1.0, float(mk.sum()) / m.sum())
            # rejected alternative: random crop keeping 70-100% of each side
            ch, cw = int(h * rng.uniform(.84, 1)), int(w * rng.uniform(.84, 1))   # ~70-100% of the area
            y0, x0 = rng.integers(0, h - ch + 1), rng.integers(0, w - cw + 1)
            kept_crop = float(m[y0:y0 + ch, x0:x0 + cw].sum()) / m.sum()
            rows.append({"type": r.defect_type, "kept_ours": kept_ours, "kept_crop": kept_crop})
    d = pd.DataFrame(rows)
    summ = {}
    for col, name in [("kept_ours", "our geometric aug"), ("kept_crop", "random crop (rejected)")]:
        summ[name] = {
            "draws": len(d),
            "min_retained_%": round(100 * d[col].min(), 1),
            "p1_retained_%": round(100 * d[col].quantile(.01), 1),
            "draws_losing_>10%_of_defect_%": round(100 * (d[col] < .9).mean(), 2),
            "draws_losing_>50%_of_defect_%": round(100 * (d[col] < .5).mean(), 2),
        }
    s = pd.DataFrame(summ).T
    print(s.to_string())
    print("\nWorst-case share of draws losing >10% of the defect, by type (our aug):")
    print((d.assign(l=d.kept_ours < .9).groupby("type").l.mean() * 100).round(2).to_dict())
    s.to_csv("docs/figures/augmentation_safety.csv")
    return d


def main():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    man = pd.read_csv(cfg["data"]["manifest"])
    # augmentation is applied to TRAIN data only; for the demo we simply use images from the trainval pool
    sample_figure(cfg, man[man.split == "trainval"], "docs/figures/augmentation_samples.png")
    safety_test(cfg, man[man.split == "trainval"])


if __name__ == "__main__":
    main()
