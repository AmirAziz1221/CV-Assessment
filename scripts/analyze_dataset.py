"""Phase 1: dataset analysis.

Run:  python scripts/analyze_dataset.py --data data/raw --out docs/figures

What it does (each step is a small function so it is easy to explain/modify):
  1. count images per class + imbalance ratio
  2. per-image properties: size, aspect ratio, channels, format, brightness, contrast, sharpness
  3. corrupt files (PIL .verify())
  4. exact duplicates (MD5), a dHash screen, and duplicate GROUPS via thumbnail correlation (src/dupes.py)
  5. "shortcut" check: can a trivial feature (width, height, brightness...) separate the classes?
  6. sample grids per class, plots
  7. defect size / location statistics from the pixel masks (drives resolution choice)
Results are written to docs/figures/*.png and docs/figures/stats.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `src` imports work from repo root
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.dupes import GROUP_THRESHOLD, duplicate_groups, thumbnail_matrix

CLASSES = ["normal", "defective"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(root: Path) -> pd.DataFrame:
    """One row per image file: path, class, defect_type (taken from the filename prefix)."""
    rows = []
    for cls in CLASSES:
        for p in sorted((root / cls).iterdir()):
            if p.suffix.lower() in EXTS:
                rows.append({"path": str(p), "label": cls, "defect_type": p.name.split("__")[0]})
    return pd.DataFrame(rows)


def dhash(img: Image.Image, size: int = 8) -> int:
    """64-bit difference hash: shrink to 9x8 grayscale, compare neighbouring pixels.
    Visually near-identical images (re-compressed, tiny shifts) get hashes within a few bits."""
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def image_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Open every image once; collect properties + hashes. Corrupt files are recorded, not crashed on."""
    recs = []
    for p in df["path"]:
        rec = {"path": p, "corrupt": False}
        try:
            with Image.open(p) as im:
                im.verify()  # cheap integrity check
            with Image.open(p) as im:  # verify() invalidates the handle, so re-open
                rec.update(width=im.width, height=im.height, mode=im.mode, format=im.format)
                gray = np.asarray(im.convert("L"), dtype=np.float32)
                rec["brightness"] = float(gray.mean())
                rec["contrast"] = float(gray.std())
                # variance of a simple Laplacian = sharpness / amount of texture
                lap = (gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:])
                rec["sharpness"] = float(lap.var())
                rec["dhash"] = dhash(im)
            rec["md5"] = hashlib.md5(Path(p).read_bytes()).hexdigest()
        except Exception as e:  # noqa: BLE001 - we want to catch anything here
            rec.update(corrupt=True, error=str(e))
        recs.append(rec)
    out = df.merge(pd.DataFrame(recs), on="path")
    out["aspect"] = out["width"] / out["height"]
    out["area"] = out["width"] * out["height"]
    return out


def find_exact_and_hash_duplicates(df: pd.DataFrame, max_hamming: int = 4):
    """Layer 1: exact dups via MD5; quick near-dup screen via Hamming distance between dhashes.
    NOTE: on smooth, low-texture images (like these tiles) dHash over-flags: many pairs within
    4 bits are NOT real copies. We therefore confirm with layer 2 (pixel correlation) below."""
    ok = df[~df.corrupt].reset_index(drop=True)
    exact = ok[ok.duplicated("md5", keep=False)].sort_values("md5")
    h = ok["dhash"].to_numpy(dtype=object)
    n_hash_pairs = sum(1 for i in range(len(h)) for j in range(i + 1, len(h))
                       if bin(h[i] ^ h[j]).count("1") <= max_hamming)
    return exact, n_hash_pairs


def shortcut_check(df: pd.DataFrame) -> dict:
    """AUC of single trivial features for predicting 'defective'.
    AUC ~0.5 = feature carries no class information. Far from 0.5 = the model could cheat on it."""
    y = (df.label == "defective").astype(int)
    out = {}
    for f in ["width", "height", "aspect", "area", "brightness", "contrast", "sharpness"]:
        out[f] = float(roc_auc_score(y, df[f]))
    return out


def sample_grid(df: pd.DataFrame, label: str, n: int, path: Path, by_type: bool = False):
    """Save an image grid. For defective we show a few of each defect type."""
    sub = df[df.label == label]
    if by_type:
        types = sorted(sub.defect_type.unique())
        per = max(1, n // len(types))
        picks = pd.concat([sub[sub.defect_type == t].sample(min(per, (sub.defect_type == t).sum()), random_state=0) for t in types])
    else:
        picks = sub.sample(n, random_state=0)
    cols = 6
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (_, r) in zip(np.ravel(axes), picks.iterrows()):
        ax.imshow(Image.open(r.path).convert("RGB"))
        ax.set_title(r.defect_type, fontsize=8)
    fig.suptitle(f"{label} samples", y=0.99)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def mask_stats(df: pd.DataFrame, mask_dir: Path, out: Path) -> pd.DataFrame:
    """Defect size/location from pixel-level masks (only available for defective images).
    area_frac = fraction of the image covered by defect -> tells us how small the defects are."""
    rec = []
    heat = np.zeros((64, 64), dtype=np.float64)
    for _, r in df[df.label == "defective"].iterrows():
        mp = mask_dir / (Path(r.path).stem + ".png")
        if not mp.exists():
            continue
        m = np.asarray(Image.open(mp).convert("L")) > 127
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        rec.append({"path": r.path, "defect_type": r.defect_type, "area_frac": m.mean(),
                    "bbox_w_frac": (xs.max() - xs.min() + 1) / m.shape[1],
                    "bbox_h_frac": (ys.max() - ys.min() + 1) / m.shape[0],
                    "cx": xs.mean() / m.shape[1], "cy": ys.mean() / m.shape[0]})
        heat += np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((64, 64))) > 127
    ms = pd.DataFrame(rec)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ms.boxplot(column="area_frac", by="defect_type", ax=ax[0])
    ax[0].set_yscale("log"); ax[0].set_title("Defect area as fraction of image (log)"); ax[0].set_xlabel("")
    fig.suptitle("")
    ax[1].imshow(heat, cmap="hot"); ax[1].set_title("Where defects occur (sum of masks)"); ax[1].axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--masks", default="data/masks")
    ap.add_argument("--out", default="docs/figures")
    a = ap.parse_args()
    root, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    df = image_properties(list_images(root))
    stats = {}

    # 1. class counts
    counts = df.label.value_counts().to_dict()
    stats["class_counts"] = counts
    stats["imbalance_ratio_normal_to_defective"] = counts["normal"] / counts["defective"]
    stats["defect_type_counts"] = df[df.label == "defective"].defect_type.value_counts().to_dict()

    # 2/3. properties + corrupt
    stats["corrupt_files"] = int(df.corrupt.sum())
    ok = df[~df.corrupt]
    stats["formats"] = ok.format.value_counts().to_dict()
    stats["modes"] = ok["mode"].value_counts().to_dict()
    stats["size_summary"] = ok[["width", "height", "aspect"]].describe().round(3).to_dict()
    stats["unique_sizes_top10"] = {f"{k[0]}x{k[1]}": v for k, v in
                                  ok.groupby(["width", "height"]).size().sort_values(ascending=False).head(10).items()}
    stats["n_unique_sizes"] = int(ok.groupby(["width", "height"]).ngroups)
    g = ok.groupby("label")[["width", "height", "brightness", "contrast", "sharpness"]].agg(["mean", "std"]).round(2)
    g.columns = [f"{a}_{b}" for a, b in g.columns]  # flatten (feature, stat) tuples so JSON works
    stats["per_class_props"] = g.to_dict("index")

    # 4. duplicates: layer 1 (md5 + dhash) and layer 2 (thumbnail correlation groups)
    exact, n_hash_pairs = find_exact_and_hash_duplicates(df)
    stats["exact_duplicate_files"] = int(len(exact))
    stats["dhash_pairs_within_4_bits"] = n_hash_pairs  # over-flags on smooth textures, see docstring
    ok = df[~df.corrupt].reset_index(drop=True)
    T = thumbnail_matrix(ok.path)
    sens = {}
    for thr in [0.95, 0.97, 0.98, 0.99, 0.995]:
        g = duplicate_groups(T, thr)
        size = np.bincount(g)[g]
        sens[str(thr)] = {
            "images_in_multi_image_groups": int((size > 1).sum()),
            "n_groups_with_2plus": int((np.bincount(g) > 1).sum()),
            "largest_group": int(np.bincount(g).max()),
            "groups_mixing_classes": int((pd.Series(ok.label.values).groupby(g).nunique() > 1).sum()),
        }
    stats["duplicate_group_sensitivity"] = sens
    GROUP_THR = GROUP_THRESHOLD  # 0.98: under-grouping = leakage, over-grouping only costs a little flexibility
    ok["dup_group"] = duplicate_groups(T, GROUP_THR)
    ok["dup_group_size"] = ok.groupby("dup_group").path.transform("size")
    stats["group_threshold_used"] = GROUP_THR
    stats["n_unique_physical_groups"] = int(ok.dup_group.nunique())
    ok[["path", "label", "defect_type", "dup_group", "dup_group_size"]].to_csv(out / "dup_groups.csv", index=False)
    stats["groups_by_class"] = ok.groupby("label").dup_group.nunique().to_dict()
    # save a few example groups for the report
    big = ok[ok.dup_group_size >= 4].dup_group.unique()[:3]
    fig, axes = plt.subplots(3, 4, figsize=(11, 8))
    for ax in axes.ravel(): ax.axis("off")
    for r, gid in enumerate(big):
        for c, (_, row) in enumerate(ok[ok.dup_group == gid].head(4).iterrows()):
            axes[r][c].imshow(Image.open(row.path).convert("L"), cmap="gray")
            axes[r][c].set_title(f"group {gid} - {row.defect_type}", fontsize=8)
    fig.suptitle(f"Examples of duplicate groups (same tile, different crop), corr>{GROUP_THR}")
    fig.tight_layout(); fig.savefig(out / "duplicate_group_examples.png", dpi=100); plt.close(fig)

    # 5. shortcut check
    stats["shortcut_auc_single_feature"] = shortcut_check(ok)

    # 6. plots
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    pd.Series(counts).reindex(CLASSES).plot.bar(ax=ax[0], color=["#4c72b0", "#c44e52"], rot=0)
    ax[0].set_title("Images per class")
    for i, v in enumerate(pd.Series(counts).reindex(CLASSES)):
        ax[0].text(i, v, str(v), ha="center", va="bottom")
    pd.Series(stats["defect_type_counts"]).plot.bar(ax=ax[1], rot=0, color="#c44e52"); ax[1].set_title("Defective images by defect type")
    fig.tight_layout(); fig.savefig(out / "class_counts.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for cls, c in zip(CLASSES, ["#4c72b0", "#c44e52"]):
        s = ok[ok.label == cls]
        ax[0].scatter(s.width, s.height, s=8, alpha=.5, label=cls, color=c)
        ax[1].hist(s.brightness, bins=40, alpha=.6, label=cls, color=c)
        ax[2].hist(np.log10(s.sharpness + 1), bins=40, alpha=.6, label=cls, color=c)
    ax[0].set_title("Width vs height (px)"); ax[1].set_title("Mean brightness"); ax[2].set_title("log10 sharpness (Laplacian var)")
    ax[0].legend(); fig.tight_layout(); fig.savefig(out / "image_properties.png", dpi=110); plt.close(fig)

    sample_grid(df, "normal", 12, out / "samples_normal.png")
    sample_grid(df, "defective", 30, out / "samples_defective.png", by_type=True)
    ms = mask_stats(df, Path(a.masks), out / "defect_size_location.png")
    if len(ms):
        stats["defect_area_frac_by_type"] = ms.groupby("defect_type").area_frac.describe()[["min", "50%", "max"]].round(5).to_dict("index")
        stats["defect_area_frac_overall_median"] = float(ms.area_frac.median())
        stats["defect_area_frac_p10"] = float(ms.area_frac.quantile(0.10))

    df.drop(columns=["dhash"]).to_csv(out / "image_properties.csv", index=False)
    (out / "stats.json").write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
