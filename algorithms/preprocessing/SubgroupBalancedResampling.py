"""
Subgroup-Balanced Resampling  (pre-processing)
교차집단 균형 리샘플링

MAF contribution — KAIST 유창동 연구실 (의료3 / ALINPneumothorax).

Idea
----
Group bias often originates in the TRAINING DATA: minority (label x protected)
cells are under-represented, so the model sees too few positive examples from
some intersectional subgroups and under-serves them. This pre-processing step
rebalances the training set by oversampling each (label x intersectional-cell)
group with replacement, with weight inversely proportional to its frequency, so
every label-by-subgroup cell is seen roughly equally during training.

It is the data-level counterpart of the weighted sampler used in the originating
chest-X-ray work, generalized to intersectional cells, and is model-agnostic
(only the resampled dataset is returned; any downstream learner can use it).

Metric reported
---------------
AIF360 standard metrics (via metrics.common_utils) on the primary attribute, and
intersectional-DEO (max-min TPR across cells), for a Logistic-Regression baseline
trained on the ORIGINAL vs the RESAMPLED training set.

Run
---
  python algorithms/preprocessing/SubgroupBalancedResampling.py
"""
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import (
    load_preproc_data_adult, load_preproc_data_german, load_preproc_data_compas)
from metrics import common_utils


class Subgroup_Balanced_Resampling:
    def __init__(self, dataset_name: str = "compas",
                 protected_attrs=None, privileged_primary: str = None,
                 target_pos_frac: float = 0.5, seed: int = 1) -> None:
        self.dataset_name = dataset_name
        defaults = {"compas": ["sex", "race"],
                    "adult":  ["sex", "race"],
                    "german": ["sex", "age"]}
        self.protected_attrs = protected_attrs or defaults[dataset_name]
        self.primary = privileged_primary or self.protected_attrs[0]
        self.target_pos_frac = target_pos_frac
        self.seed = seed
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        if self.dataset_name == 'adult':
            self.dataset_orig = load_preproc_data_adult(self.protected_attrs)
        elif self.dataset_name == 'german':
            self.dataset_orig = load_preproc_data_german(self.protected_attrs)
        elif self.dataset_name == 'compas':
            self.dataset_orig = load_preproc_data_compas(self.protected_attrs)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_train, self.dataset_test = self.dataset_orig.split([0.7], shuffle=True, seed=1)
        self.privileged_groups = [{self.primary: 1}]
        self.unprivileged_groups = [{self.primary: 0}]
        self.fav = self.dataset_orig.favorable_label
        self.attr_idx = {n: i for i, n in enumerate(self.dataset_orig.protected_attribute_names)}

    def _cell_ids(self, dataset):
        cols = [dataset.protected_attributes[:, self.attr_idx[a]] for a in self.protected_attrs]
        return [tuple(row) for row in np.stack(cols, axis=1)]

    def balanced_indices(self):
        """Oversample (label x intersectional-cell) groups to equal size."""
        rng = np.random.default_rng(self.seed)
        y = self.dataset_train.labels.ravel()
        cells = self._cell_ids(self.dataset_train)
        groups = {}
        for i, (lab, cell) in enumerate(zip(y, cells)):
            groups.setdefault((lab, cell), []).append(i)
        target = max(len(v) for v in groups.values())   # size of the largest group
        idx = []
        for g, members in groups.items():
            idx.extend(members)                          # keep originals
            need = target - len(members)                 # oversample the rest
            if need > 0:
                idx.extend(rng.choice(members, size=need, replace=True).tolist())
        rng.shuffle(idx)
        return np.array(idx)

    def _fit_eval(self, train_ds):
        scaler = StandardScaler()
        X = scaler.fit_transform(train_ds.features)
        y = train_ds.labels.ravel()
        lmod = LogisticRegression(max_iter=1000)
        lmod.fit(X, y)
        test_pred = self.dataset_test.copy(deepcopy=True)
        test_pred.labels = lmod.predict(scaler.transform(self.dataset_test.features)).reshape(-1, 1)
        return test_pred

    def intersectional_deo(self, dataset_true, pred):
        yt = dataset_true.labels.ravel(); yp = pred.labels.ravel()
        cells = self._cell_ids(dataset_true); tprs = []
        for cell in set(cells):
            m = np.array([c == cell for c in cells]); pos = m & (yt == self.fav)
            if pos.sum() == 0:
                continue
            tprs.append(float((yp[pos] == self.fav).mean()))
        return (max(tprs) - min(tprs)) if len(tprs) >= 2 else float('nan')

    def run(self):
        # BEFORE: baseline on original training data
        test_orig = self._fit_eval(self.dataset_train)
        metrics_orig = common_utils.compute_metrics(
            self.dataset_test, test_orig,
            self.unprivileged_groups, self.privileged_groups, disp=False)
        metrics_orig["intersectional_DEO"] = self.intersectional_deo(self.dataset_test, test_orig)

        # AFTER: baseline on subgroup-balanced (resampled) training data
        idx = self.balanced_indices()
        train_bal = self.dataset_train.subset(list(idx))
        test_bal = self._fit_eval(train_bal)
        metrics_transf = common_utils.compute_metrics(
            self.dataset_test, test_bal,
            self.unprivileged_groups, self.privileged_groups, disp=False)
        metrics_transf["intersectional_DEO"] = self.intersectional_deo(self.dataset_test, test_bal)
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    clf = Subgroup_Balanced_Resampling(dataset_name="compas", protected_attrs=["sex", "race"])
    m_orig, m_transf = clf.run()
    keys = ["accuracy", "Balanced accuracy", "equal_opportunity_difference",
            "average_abs_odds_difference", "intersectional_DEO"]
    print(f"{'metric':32s}{'BEFORE(original)':>18s}{'AFTER(balanced)':>18s}")
    for k in keys:
        print(f"{k:32s}{m_orig[k]:>18.4f}{m_transf[k]:>18.4f}")
