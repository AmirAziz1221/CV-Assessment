"""Phase 2, step 2: choose the input resolution and resize mode WITH DATA, before training anything.

Question: after resizing to R x R, how many pixels does a defect still have?
  * 'stretch'   : the whole image is squashed to R x R -> defect side = sqrt(area_fraction) * R
  * 'letterbox' : long side -> R, aspect kept          -> defect side = sqrt(area_fraction * w * h) * scale
We use the 392 pixel masks (defect area) and the true image sizes. 'side' = sqrt(defect area in pixels),
i.e. the width of a square with the same area. A defect with a side of only a few pixels is almost
invisible to a CNN (the first conv layers and the stride-32 backbone would blur it away).

Run:  python scripts/resolution_study.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RES = [224, 256, 320, 384, 448]


def main():
    man = pd.read_csv("data/splits.csv")
    rec = []
    for _, r in man[man.label == "defective"].iterrows():
        mp = Path("data/masks") / (Path(r.path).stem + ".png")
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L")) > 127
        if m.sum() == 0:
            continue
        h, w = m.shape
        rec.append({"type": r.defect_type, "w": w, "h": h, "area_frac": m.mean()})
    d = pd.DataFrame(rec)
    if d.empty:
        print("no masks found - skipping study"); return

    out = []
    for R in RES:
        scale = np.minimum(R / d.w, R / d.h)
        letter = np.sqrt(d.area_frac * d.w * d.h) * scale          # defect side in px, letterbox
        stretch = np.sqrt(d.area_frac) * R                          # defect side in px, stretch
        canvas_used = (d.w * scale * d.h * scale) / (R * R)         # fraction of canvas holding real image
        for name, side in [("letterbox", letter), ("stretch", stretch)]:
            out.append({"R": R, "mode": name,
                        "median_side_px": side.median(), "p10_side_px": side.quantile(.10),
                        "pct_defects_below_6px": 100 * (side < 6).mean(),
                        "pct_defects_below_10px": 100 * (side < 10).mean(),
                        "canvas_used_pct": 100 * (canvas_used.mean() if name == "letterbox" else 1.0),
                        "relative_cost": (R / 224) ** 2})
    t = pd.DataFrame(out).round(1)
    t.to_csv("docs/figures/resolution_study.csv", index=False)
    print(t.to_string(index=False))

    # by defect type at the two main candidates
    print("\nMedian defect side (px) by type, letterbox:")
    for R in (224, 320):
        s = np.minimum(R / d.w, R / d.h)
        print(R, (np.sqrt(d.area_frac * d.w * d.h) * s).groupby(d.type).median().round(1).to_dict())

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for mode, c in [("letterbox", "#4c72b0"), ("stretch", "#c44e52")]:
        s = t[t["mode"] == mode]
        ax[0].plot(s.R, s.p10_side_px, "o-", color=c, label=mode)
        ax[1].plot(s.R, s.pct_defects_below_10px, "o-", color=c, label=mode)
    ax[0].set_title("Size of the smallest-10% defects (px)"); ax[0].set_xlabel("input size R")
    ax[1].set_title("% of defects with side < 10 px"); ax[1].set_xlabel("input size R"); ax[0].legend()
    fig.tight_layout(); fig.savefig("docs/figures/resolution_study.png", dpi=110)


if __name__ == "__main__":
    main()
