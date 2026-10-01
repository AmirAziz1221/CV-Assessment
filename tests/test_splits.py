"""Tests for group-aware splitting. Uses synthetic data, so no dataset is needed."""
import numpy as np
import pandas as pd

from src.splits import assign_splits


def make_fake(n_groups=200, seed=0):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(1, 6, n_groups)                       # 1-5 crops per 'tile'
    groups = np.repeat(np.arange(n_groups), sizes)
    # each tile has one defect type (like real data); 'free' is most common
    types = rng.choice(["free", "free", "free", "crack", "blowhole", "fray"], n_groups)
    return types[groups], groups


def test_no_group_in_two_splits():
    y, g = make_fake()
    split, fold = assign_splits(y, g, seed=1)
    df = pd.DataFrame({"g": g, "split": split, "fold": fold})
    assert (df.groupby("g").split.nunique() == 1).all()
    tv = df[df.split == "trainval"]
    assert (tv.groupby("g").fold.nunique() == 1).all()


def test_sizes_and_folds():
    y, g = make_fake()
    split, fold = assign_splits(y, g, seed=1)
    frac_test = (split == "test").mean()
    assert 0.10 < frac_test < 0.19                              # ~1/7
    assert set(fold[split == "test"]) == {-1}
    assert set(fold[split == "trainval"]) == {0, 1, 2, 3, 4}


def test_every_type_in_test_and_every_fold():
    y, g = make_fake()
    split, fold = assign_splits(y, g, seed=1)
    for t in set(y):
        assert t in set(y[split == "test"])
        for k in range(5):
            assert t in set(y[(split == "trainval") & (fold == k)])


def test_reproducible_and_seed_sensitive():
    y, g = make_fake()
    a = assign_splits(y, g, seed=7)
    b = assign_splits(y, g, seed=7)
    c = assign_splits(y, g, seed=8)
    assert (a[0] == b[0]).all() and (a[1] == b[1]).all()
    assert not (a[0] == c[0]).all()
