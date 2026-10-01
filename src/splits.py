"""Group-aware, stratified, reproducible splits.

Design (decided from the Phase 1 findings):
  * The unit we split is the DUPLICATE GROUP (all crops of one physical tile), never the single image.
    -> no tile can appear in two splits -> no leakage between train / val / test.
  * Stratify on the defect TYPE (6 values: free + 5 defect types), so every split contains every
    defect type, even the rare 'fray'.
  * Step 1: carve out a locked TEST set (1 of 7 folds ~ 14.3% ~ the 15% asked for).
  * Step 2: split the remaining ~85% into 5 grouped folds. Each fold is ~17% of the data, so
    "train on 4 folds, validate on 1" is ~68% / 17% / 15% = the requested 70/15/15.
    The 5 folds give k-fold cross-validation for free (a mean +- std instead of one noisy number).
  * Fixed seed + the result is saved as a CSV manifest so everything is reproducible.
"""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def assign_splits(strat_labels, groups, seed: int = 42, n_test_splits: int = 7, n_cv_folds: int = 5):
    """Return (split, fold) arrays. split in {'test','trainval'}; fold in {0..n_cv_folds-1} or -1 for test."""
    strat_labels, groups = np.asarray(strat_labels), np.asarray(groups)
    n = len(strat_labels)
    idx = np.arange(n)
    split = np.full(n, "trainval", dtype=object)
    fold = np.full(n, -1, dtype=int)

    # Step 1: locked test set = the first held-out fold of a 7-fold grouped split
    outer = StratifiedGroupKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    _, test_idx = next(iter(outer.split(idx, strat_labels, groups)))
    split[test_idx] = "test"

    # Step 2: the rest -> n_cv_folds grouped, stratified folds
    tv = np.where(split == "trainval")[0]
    inner = StratifiedGroupKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)
    for k, (_, val_local) in enumerate(inner.split(tv, strat_labels[tv], groups[tv])):
        fold[tv[val_local]] = k
    return split, fold
