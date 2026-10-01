"""Data loading, preprocessing and augmentation (Phase 2).

Pipeline for ONE image, in order (each step has a reason, see docs/data_prep.md):

  TRAIN:  load grayscale -> 3 channels
          -> geometric aug (flip, small rotation, small shift)      [native resolution]
          -> photometric aug (brightness/contrast, light blur)
          -> resize to the model resolution (aspect-preserving 'letterbox' or plain 'stretch')
          -> float [0,1] -> light gaussian noise -> ImageNet normalisation -> zero-pad to square
  EVAL:   same, minus all augmentation.

Padding is applied AFTER normalisation with value 0, which equals "the ImageNet mean colour",
so the padding is neutral rather than a bright/dark border the network could latch onto.
"""
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["normal", "defective"]  # label_id 0 / 1 ; 'defective' is the POSITIVE class


# ----------------------------------------------------------------------------- small custom transforms
class FitInside(torch.nn.Module):
    """Resize so the LONG side equals `size`, keeping the aspect ratio (no distortion)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, img):
        _, h, w = img.shape
        s = self.size / max(h, w)
        return v2.functional.resize(img, [max(1, round(h * s)), max(1, round(w * s))], antialias=True)


class PadToSquare(torch.nn.Module):
    """Centre the image on a size x size canvas filled with 0 (= ImageNet mean after normalisation)."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, img):
        _, h, w = img.shape
        top, left = (self.size - h) // 2, (self.size - w) // 2
        return v2.functional.pad(img, [left, top, self.size - w - left, self.size - h - top], fill=0)


class AddGaussianNoise(torch.nn.Module):
    """Sensor-like noise on a [0,1] float image, applied with probability p."""

    def __init__(self, std: float = 0.02, p: float = 0.3):
        super().__init__()
        self.std, self.p = std, p

    def forward(self, img):
        if torch.rand(1).item() < self.p:
            img = (img + torch.randn_like(img) * self.std).clamp(0, 1)
        return img


# ----------------------------------------------------------------------------- transform builders
def geometric_aug(cfg: dict) -> v2.Compose:
    """Geometric augmentations only. Kept separate because v2 applies them identically to an
    image AND its defect mask -> lets us test that augmentation never erases a defect.

    Lessons from scripts/augmentation_check.py:
      * a small shift / RandomAffine pushed edge defects out of the frame (up to 35% of 'break' draws lost
        >10% of the defect) -> removed. Random crops were even worse -> not used.
      * rotation uses expand=True: the canvas grows to hold the rotated image, so NOTHING is cut off
        (the new corners are filled with black, like the near-black tile borders already in the data).
    """
    a = cfg["augment"]
    return v2.Compose([
        v2.RandomHorizontalFlip(a["hflip_p"]),
        v2.RandomVerticalFlip(a["vflip_p"]),
        v2.RandomRotation(degrees=a["rotate_deg"], expand=True, fill=0),
    ])


def photometric_aug(cfg: dict) -> v2.Compose:
    a = cfg["augment"]
    return v2.Compose([
        v2.ColorJitter(brightness=a["brightness"], contrast=a["contrast"]),
        v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, a["blur_sigma_max"]))], p=a["blur_p"]),
    ])


def resize_step(cfg: dict):
    s, mode = cfg["image_size"], cfg["resize_mode"]
    if mode == "letterbox":
        return FitInside(s)
    if mode == "stretch":
        return v2.Resize((s, s), antialias=True)
    raise ValueError(f"unknown resize_mode {mode!r} (use 'letterbox' or 'stretch')")


def build_transform(cfg: dict, train: bool):
    steps = [v2.ToImage()]  # PIL (RGB) -> uint8 tensor [3,H,W]
    if train:
        steps += [geometric_aug(cfg), photometric_aug(cfg)]
    steps += [resize_step(cfg), v2.ToDtype(torch.float32, scale=True)]
    if train:
        steps += [AddGaussianNoise(cfg["augment"]["noise_std"], cfg["augment"]["noise_p"])]
    steps += [v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if cfg["resize_mode"] == "letterbox":
        steps += [PadToSquare(cfg["image_size"])]
    return v2.Compose(steps)


# ----------------------------------------------------------------------------- dataset + imbalance helpers
class DefectDataset(Dataset):
    """Reads rows of the split manifest (data/splits.csv). Returns (image_tensor, label_id)."""

    def __init__(self, manifest: pd.DataFrame, transform):
        self.df = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")  # grayscale JPEG -> replicate to 3 channels
        return self.transform(img), int(row["label_id"])


def select(manifest: pd.DataFrame, role: str, fold: int = 0) -> pd.DataFrame:
    """role: 'train' (trainval minus `fold`), 'val' (== `fold`), 'test' (locked, final evaluation only)."""
    if role == "test":
        return manifest[manifest.split == "test"]
    tv = manifest[manifest.split == "trainval"]
    return tv[tv.fold != fold] if role == "train" else tv[tv.fold == fold]


def class_weights(labels) -> torch.Tensor:
    """Inverse-frequency weights for the loss: w_c = N / (n_classes * n_c). Rarer class -> larger weight."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=2)
    return torch.tensor(len(labels) / (2.0 * counts), dtype=torch.float32)


def balanced_sampler(labels) -> WeightedRandomSampler:
    """Draw samples with prob ~ 1/class_frequency -> batches are ~50/50. Combined with augmentation,
    each repeat of a minority image is a different variant (so it is 'oversampling with augmentation')."""
    labels = np.asarray(labels)
    w = (1.0 / np.bincount(labels))[labels]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)
