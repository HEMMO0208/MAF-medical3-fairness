"""
Multi-seed robustness check: rerun an algorithm with different split seeds and
report mean +/- std of each metric (before vs after), plus per-cell positives/TPR
for the image algorithm.

  python tools/multiseed.py --algo itc        --seeds 1 2 3 4 5   # ① tabular (COMPAS)
  python tools/multiseed.py --algo sbr        --seeds 1 2 3 4 5   # ② tabular (COMPAS)
  python tools/multiseed.py --algo itc --dataset adult              # ① on Adult (or german)
  python tools/multiseed.py --algo itc_image  --seeds 1 2 3 4 5   # ① image   (SIIM-ACR)
"""
import os
import sys
import json
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

KEYS = ["accuracy", "Balanced accuracy", "true_positive_rate",
        "equal_opportunity_difference", "average_abs_odds_difference", "intersectional_DEO"]


DEFAULT_ATTRS = {'compas': ['sex', 'race'], 'adult': ['sex', 'race'], 'german': ['sex', 'age']}


def make(algo, seed, dataset='compas'):
    if algo == 'itc':
        from algorithms.postprocessing.IntersectionalThresholdCalibration import Intersectional_Threshold_Calibration
        return Intersectional_Threshold_Calibration(dataset_name=dataset, protected_attrs=DEFAULT_ATTRS[dataset], seed=seed)
    if algo == 'sbr':
        from algorithms.preprocessing.SubgroupBalancedResampling import Subgroup_Balanced_Resampling
        return Subgroup_Balanced_Resampling(dataset_name=dataset, protected_attrs=DEFAULT_ATTRS[dataset], seed=seed)
    if algo in ('itc_image', 'itc_image_pixel'):
        from algorithms.postprocessing.IntersectionalThresholdCalibration_Image import Intersectional_Threshold_Calibration_Image
        if algo == 'itc_image':
            return Intersectional_Threshold_Calibration_Image(dataset_name='siim', protected='sex', seed=seed)
        # guide-literal variant: 64x64 pixels, 60/20/20 split
        return Intersectional_Threshold_Calibration_Image(dataset_name='siim', protected='sex', seed=seed,
                                                          feature='pixel', train_frac=0.6)
    raise ValueError(algo)


def per_cell(clf):
    """Positives and TPR per intersectional cell on the test split (image algo)."""
    valid_pred, test_pred = clf.baseline_fit()
    clf.fit_thresholds(valid_pred)
    before = clf.apply_thresholds(test_pred, clf.dataset_test, per_cell=False)
    after = clf.apply_thresholds(test_pred, clf.dataset_test, per_cell=True)
    y = clf.dataset_test.labels.ravel()
    cells = clf._cell_ids(clf.dataset_test)
    out = {}
    for c in sorted(set(cells)):
        m = np.array([x == c for x in cells]) & (y == clf.fav)
        name = ' x '.join(f'{a}={int(v)}' for a, v in zip(clf.protected_attrs, c))
        out[name] = dict(n_pos=int(m.sum()),
                         TPR_before=round(float((before.labels.ravel()[m] == clf.fav).mean()), 4) if m.sum() else None,
                         TPR_after=round(float((after.labels.ravel()[m] == clf.fav).mean()), 4) if m.sum() else None,
                         threshold=round(float(clf.cell_thresholds.get(c, clf.global_thr)), 3))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--algo', required=True, choices=['itc', 'sbr', 'itc_image', 'itc_image_pixel'])
    ap.add_argument('--dataset', default='compas', choices=['compas', 'adult', 'german'],
                    help='tabular dataset for itc / sbr')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--out', default=None, help='optional JSON output path')
    a = ap.parse_args()

    rec = {'algo': a.algo, 'dataset': a.dataset, 'seeds': a.seeds, 'per_seed': [], 'per_cell_seed': {}}
    for s in a.seeds:
        clf = make(a.algo, s, a.dataset)
        mo, mt = clf.run()
        row = {'seed': s, 'before': {k: float(mo[k]) for k in KEYS}, 'after': {k: float(mt[k]) for k in KEYS}}
        rec['per_seed'].append(row)
        print(f"[seed {s}] interDEO {mo['intersectional_DEO']:.3f}->{mt['intersectional_DEO']:.3f} | "
              f"EOD {mo['equal_opportunity_difference']:+.3f}->{mt['equal_opportunity_difference']:+.3f} | "
              f"bal_acc {mo['Balanced accuracy']:.3f}->{mt['Balanced accuracy']:.3f}", flush=True)
        if a.algo.startswith('itc_image'):
            rec['per_cell_seed'][s] = per_cell(clf)

    print(f"\n=== {len(a.seeds)} seeds: mean +/- std ===")
    rec['summary'] = {}
    for k in KEYS:
        b = np.array([r['before'][k] for r in rec['per_seed']])
        f = np.array([r['after'][k] for r in rec['per_seed']])
        rec['summary'][k] = dict(before_mean=float(b.mean()), before_std=float(b.std()),
                                 after_mean=float(f.mean()), after_std=float(f.std()))
        print(f"{k:32s} before {b.mean():.4f} +/- {b.std():.4f}   after {f.mean():.4f} +/- {f.std():.4f}")
    if a.algo.startswith('itc_image'):
        print(f"\n=== per-cell positives / TPR (seed {a.seeds[0]}) ===")
        print(json.dumps(rec['per_cell_seed'][a.seeds[0]], indent=2))
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(rec, f, indent=2)
        print(f"\nsaved {a.out}")


if __name__ == '__main__':
    main()
