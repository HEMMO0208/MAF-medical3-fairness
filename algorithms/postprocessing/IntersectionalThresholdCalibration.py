"""
Intersectional Equalized-Opportunity Threshold Calibration  (post-processing)
교차집단 등가기회 임계값 보정

MAF contribution — KAIST 유창동 연구실 (의료3 / ALINPneumothorax).

Idea
----
Standard group-fairness post-processing (e.g. Reject-Option, single-attribute
equalized-odds) corrects bias along ONE protected axis at a time. In practice a
model can look fair on sex alone AND on age alone, yet badly under-serve an
INTERSECTIONAL subgroup (e.g. older males). This algorithm fixes a trained
classifier's bias at the intersectional (e.g. sex x race, or age x sex) level,
without retraining:

  1. Train any baseline classifier -> prediction scores.
  2. On a held-out VALIDATION split, pick a per-(intersectional-cell) decision
     threshold so that every cell's TPR (true-positive rate) is pulled toward a
     common target TPR  ->  equal opportunity across all cells.
  3. Sparse cells (too few positives to estimate a stable threshold) fall back
     to the global threshold (honest, avoids over-fitting tiny cells).
  4. Apply the frozen per-cell thresholds to the TEST split.

It is model-agnostic (operates only on scores + protected attributes + labels),
so it plugs onto tabular AIF360 datasets and, identically, onto image models via
their predicted scores (demonstrated on chest X-ray in the originating work).

Metric reported
---------------
- AIF360 standard metrics (via metrics.common_utils) on the PRIMARY attribute.
- intersectional-DEO = max_cell TPR - min_cell TPR  (lower = fairer), before vs
  after calibration — the quantity this algorithm directly minimizes.

Run
---
  python algorithms/postprocessing/IntersectionalThresholdCalibration.py
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


# minimum #positives in the validation half of a cell to fit its own threshold;
# sparser cells fall back to the global threshold.
MIN_POS = 20


class Intersectional_Threshold_Calibration:
    def __init__(self, dataset_name: str = "compas",
                 protected_attrs=None, privileged_primary: str = None, seed: int = 1) -> None:
        self.dataset_name = dataset_name
        self.seed = seed
        # protected_attrs: the axes whose INTERSECTION defines the subgroups.
        # defaults give a 2-axis intersection per dataset.
        defaults = {"compas": ["sex", "race"],
                    "adult":  ["sex", "race"],
                    "german": ["sex", "age"]}
        self.protected_attrs = protected_attrs or defaults[dataset_name]
        # primary axis is used for the AIF360 single-attribute standard metrics
        self.primary = privileged_primary or self.protected_attrs[0]
        self.load_and_preprocess_data()

    # ----------------------------------------------------------------- data
    def load_and_preprocess_data(self):
        if self.dataset_name == 'adult':
            self.dataset_orig = load_preproc_data_adult(self.protected_attrs)
        elif self.dataset_name == 'german':
            self.dataset_orig = load_preproc_data_german(self.protected_attrs)
        elif self.dataset_name == 'compas':
            self.dataset_orig = load_preproc_data_compas(self.protected_attrs)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_train, dataset_vt = self.dataset_orig.split([0.6], shuffle=True, seed=self.seed)
        self.dataset_valid, self.dataset_test = dataset_vt.split([0.5], shuffle=True, seed=self.seed)

        self.privileged_groups = [{self.primary: 1}]
        self.unprivileged_groups = [{self.primary: 0}]

        self.fav = self.dataset_orig.favorable_label
        self.unfav = self.dataset_orig.unfavorable_label
        self.attr_idx = {n: i for i, n in enumerate(self.dataset_orig.protected_attribute_names)}

    # --------------------------------------------------------------- helpers
    def _cell_ids(self, dataset):
        """Map each sample to an intersectional-cell id (tuple of its protected
        attribute values across self.protected_attrs)."""
        cols = [dataset.protected_attributes[:, self.attr_idx[a]] for a in self.protected_attrs]
        cells = np.stack(cols, axis=1)
        ids = [tuple(row) for row in cells]
        return ids

    def _tpr(self, y_true, y_pred):
        pos = (y_true == self.fav)
        if pos.sum() == 0:
            return np.nan
        return float((y_pred[pos] == self.fav).mean())

    def _labels_at(self, scores, thr):
        out = np.where(scores >= thr, self.fav, self.unfav)
        return out

    # ------------------------------------------------------------- baseline
    def baseline_fit(self):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(self.dataset_train.features)
        y_tr = self.dataset_train.labels.ravel()
        lmod = LogisticRegression(max_iter=1000)
        lmod.fit(X_tr, y_tr)
        fav_col = np.where(lmod.classes_ == self.fav)[0][0]

        valid_pred = self.dataset_valid.copy(deepcopy=True)
        test_pred = self.dataset_test.copy(deepcopy=True)
        valid_pred.scores = lmod.predict_proba(scaler.transform(self.dataset_valid.features))[:, fav_col].reshape(-1, 1)
        test_pred.scores = lmod.predict_proba(scaler.transform(self.dataset_test.features))[:, fav_col].reshape(-1, 1)
        return valid_pred, test_pred

    def _global_threshold(self, valid_pred):
        """Threshold on validation maximising balanced accuracy (fairness-free)."""
        sc = valid_pred.scores.ravel()
        yt = self.dataset_valid.labels.ravel()
        best_t, best_ba = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            yp = self._labels_at(sc, t)
            tpr = self._tpr(yt, yp)
            tnr_mask = (yt == self.unfav)
            tnr = float((yp[tnr_mask] == self.unfav).mean()) if tnr_mask.sum() else np.nan
            ba = 0.5 * (tpr + tnr)
            if ba > best_ba:
                best_ba, best_t = ba, t
        return best_t

    # ----------------------------------------------- the calibration (fit)
    def fit_thresholds(self, valid_pred):
        """Per-cell equal-opportunity thresholds fit on validation."""
        self.global_thr = self._global_threshold(valid_pred)
        sc = valid_pred.scores.ravel()
        yt = self.dataset_valid.labels.ravel()
        cells = self._cell_ids(self.dataset_valid)

        # common target TPR = overall TPR at the global threshold on validation
        target_tpr = self._tpr(yt, self._labels_at(sc, self.global_thr))

        thr = {}
        grid = np.linspace(0.01, 0.99, 99)
        for cell in set(cells):
            m = np.array([c == cell for c in cells])
            pos = m & (yt == self.fav)
            if pos.sum() < MIN_POS:
                thr[cell] = self.global_thr            # sparse-cell fallback
                continue
            # choose threshold whose cell-TPR is closest to target_tpr
            best_t, best_gap = self.global_thr, 1e9
            for t in grid:
                yp = self._labels_at(sc[m], t)
                cell_tpr = float((yp[yt[m] == self.fav] == self.fav).mean())
                gap = abs(cell_tpr - target_tpr)
                if gap < best_gap:
                    best_gap, best_t = gap, t
            thr[cell] = best_t
        self.cell_thresholds = thr
        self.target_tpr = target_tpr
        return thr

    def apply_thresholds(self, pred, dataset_ref, per_cell=True):
        """Return a copy of `pred` with labels set by per-cell (or global) thresholds."""
        out = pred.copy(deepcopy=True)
        sc = pred.scores.ravel()
        if per_cell:
            cells = self._cell_ids(dataset_ref)
            yp = np.empty_like(sc)
            for i, cell in enumerate(cells):
                t = self.cell_thresholds.get(cell, self.global_thr)
                yp[i] = self.fav if sc[i] >= t else self.unfav
        else:
            yp = self._labels_at(sc, self.global_thr)
        out.labels = yp.reshape(-1, 1)
        return out

    # --------------------------------------------------- intersectional DEO
    def intersectional_deo(self, dataset_true, pred):
        yt = dataset_true.labels.ravel()
        yp = pred.labels.ravel()
        cells = self._cell_ids(dataset_true)
        tprs = []
        for cell in set(cells):
            m = np.array([c == cell for c in cells])
            pos = m & (yt == self.fav)
            if pos.sum() == 0:
                continue
            tprs.append(float((yp[pos] == self.fav).mean()))
        if len(tprs) < 2:
            return float('nan')
        return max(tprs) - min(tprs)

    # ------------------------------------------------------------------ run
    def run(self):
        valid_pred, test_pred = self.baseline_fit()
        self.fit_thresholds(valid_pred)

        # BEFORE: single global threshold
        test_global = self.apply_thresholds(test_pred, self.dataset_test, per_cell=False)
        metrics_orig = common_utils.compute_metrics(
            self.dataset_test, test_global,
            self.unprivileged_groups, self.privileged_groups, disp=False)
        metrics_orig["intersectional_DEO"] = self.intersectional_deo(self.dataset_test, test_global)

        # AFTER: per-(intersectional-cell) equal-opportunity thresholds
        test_cal = self.apply_thresholds(test_pred, self.dataset_test, per_cell=True)
        metrics_transf = common_utils.compute_metrics(
            self.dataset_test, test_cal,
            self.unprivileged_groups, self.privileged_groups, disp=False)
        metrics_transf["intersectional_DEO"] = self.intersectional_deo(self.dataset_test, test_cal)
        return metrics_orig, metrics_transf


if __name__ == "__main__":
    clf = Intersectional_Threshold_Calibration(dataset_name="compas",
                                               protected_attrs=["sex", "race"])
    m_orig, m_transf = clf.run()
    keys = ["accuracy", "Balanced accuracy", "equal_opportunity_difference",
            "average_abs_odds_difference", "intersectional_DEO"]
    print(f"Intersectional cells & thresholds: {clf.cell_thresholds}")
    print(f"Target TPR (equal opportunity) = {clf.target_tpr:.3f}\n")
    print(f"{'metric':32s}{'BEFORE(global)':>16s}{'AFTER(per-cell)':>18s}")
    for k in keys:
        print(f"{k:32s}{m_orig[k]:>16.4f}{m_transf[k]:>18.4f}")
