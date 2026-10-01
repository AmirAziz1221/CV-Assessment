"""Tests for preprocessing / augmentation / imbalance helpers (synthetic images, no dataset needed)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from PIL import Image
from torchvision import tv_tensors

from src.data import DefectDataset, balanced_sampler, build_transform, class_weights, geometric_aug, select

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))["data"]


def fake_img(w, h, value=128):
    return Image.fromarray(np.full((h, w, 3), value, dtype=np.uint8))


@pytest.mark.parametrize("mode", ["stretch", "letterbox"])
@pytest.mark.parametrize("size", [(300, 300), (600, 240), (110, 400)])   # square, wide, tall
def test_eval_transform_shape(mode, size):
    cfg = {**CFG, "resize_mode": mode, "image_size": 64}
    out = build_transform(cfg, train=False)(fake_img(*size))
    assert out.shape == (3, 64, 64) and out.dtype == torch.float32


def test_normalisation_uses_imagenet_stats():
    cfg = {**CFG, "resize_mode": "stretch", "image_size": 32}
    out = build_transform(cfg, train=False)(fake_img(50, 50, value=255))   # white -> (1-mean)/std
    assert torch.allclose(out[0], torch.full((32, 32), (1 - 0.485) / 0.229), atol=1e-3)


def test_letterbox_keeps_aspect_and_pads_with_zero():
    cfg = {**CFG, "resize_mode": "letterbox", "image_size": 64}
    out = build_transform(cfg, train=False)(fake_img(200, 50, value=255))  # wide image
    assert out[:, 0, :].abs().max() == 0 and out[:, -1, :].abs().max() == 0  # top/bottom rows are padding
    assert out[:, 32, :].abs().min() > 0                                     # middle rows hold the image


def test_train_transform_is_random_but_shape_stable():
    cfg = {**CFG, "resize_mode": "stretch", "image_size": 48}
    tf = build_transform(cfg, train=True)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 120, 3), dtype=np.uint8))
    outs = [tf(img) for _ in range(5)]
    assert all(o.shape == (3, 48, 48) for o in outs)
    assert not all(torch.equal(outs[0], o) for o in outs[1:])


def test_geometric_aug_never_erases_a_defect():
    """A small defect right at the image border must survive flips + rotation (expand=True)."""
    geo = geometric_aug(CFG)
    img = tv_tensors.Image(torch.zeros(3, 100, 100, dtype=torch.uint8))
    m = torch.zeros(100, 100, dtype=torch.uint8)
    m[0:6, 0:6] = 1                                                          # defect in the corner
    mask = tv_tensors.Mask(m)
    torch.manual_seed(0)
    kept = [float(geo(img, mask)[1].sum()) / m.sum() for _ in range(100)]
    assert min(kept) > 0.85


def test_class_weights_and_sampler():
    labels = np.array([0] * 80 + [1] * 20)
    w = class_weights(labels)
    assert w[1] > w[0] and torch.isclose(w[1] / w[0], torch.tensor(4.0))
    drawn = np.array([labels[i] for i in balanced_sampler(labels)])
    assert 0.35 < drawn.mean() < 0.65                                        # ~50% defective after balancing


def test_dataset_and_select(tmp_path):
    rows = []
    for i in range(10):
        p = tmp_path / f"{i}.jpg"
        fake_img(40, 30).save(p)
        rows.append({"path": str(p), "label_id": i % 2, "split": "test" if i < 2 else "trainval",
                     "fold": -1 if i < 2 else (i - 2) % 5})
    man = pd.DataFrame(rows)
    train, val, test = select(man, "train", 0), select(man, "val", 0), select(man, "test")
    assert set(train.index).isdisjoint(val.index) and len(test) == 2 and (val.fold == 0).all()
    x, y = DefectDataset(train, build_transform({**CFG, "image_size": 32}, train=False))[0]
    assert x.shape == (3, 32, 32) and y in (0, 1)


@pytest.mark.skipif(not Path("data/splits.csv").exists(), reason="manifest not generated yet")
def test_real_manifest_has_no_group_leakage():
    man = pd.read_csv("data/splits.csv")
    assert (man.groupby("group").split.nunique() == 1).all()
    assert set(man.label) == {"normal", "defective"}
