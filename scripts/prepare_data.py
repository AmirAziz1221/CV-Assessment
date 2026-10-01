"""Step 0: download the public Magnetic Tile Surface Defects dataset and arrange it as
    data/raw/normal/*.jpg      (MT_Free)
    data/raw/defective/*.jpg   (MT_Blowhole, MT_Break, MT_Crack, MT_Fray, MT_Uneven)
    data/masks/*.png           (pixel masks of the defective images, used only for analysis / Grad-CAM checks)
    data/source_labels.csv     (filename, label, defect_type)

The defect type is kept as a filename prefix ("crack__exp1_num_123.jpg") so error analysis can group by type.

To use YOUR OWN data instead: skip this script and put images in data/raw/normal and data/raw/defective.
(Masks are optional; scripts that need them skip gracefully when they are missing.)

Run:  python scripts/prepare_data.py
"""
import csv
import shutil
import subprocess
from pathlib import Path

REPO = "https://github.com/abin24/Magnetic-tile-defect-datasets..git"  # the trailing '..' is part of the real repo name
DL = Path("data/_download/mt")


def main():
    if not DL.exists():
        DL.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(DL)], check=True)

    for d in ["data/raw/normal", "data/raw/defective", "data/masks"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    rows = []
    for folder in sorted(DL.glob("MT_*")):
        kind = folder.name[3:].lower()                    # free, blowhole, break, crack, fray, uneven
        label = "normal" if kind == "free" else "defective"
        for jpg in sorted((folder / "Imgs").glob("*.jpg")):  # the .png files next to them are masks
            new_name = f"{kind}__{jpg.name}"
            shutil.copy(jpg, Path("data/raw") / label / new_name)
            mask = jpg.with_suffix(".png")
            if label == "defective" and mask.exists():
                shutil.copy(mask, Path("data/masks") / new_name.replace(".jpg", ".png"))
            rows.append((new_name, label, kind))

    with open("data/source_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "defect_type"])
        w.writerows(rows)
    n_norm = sum(r[1] == "normal" for r in rows)
    print(f"done: {n_norm} normal, {len(rows) - n_norm} defective images")


if __name__ == "__main__":
    main()
