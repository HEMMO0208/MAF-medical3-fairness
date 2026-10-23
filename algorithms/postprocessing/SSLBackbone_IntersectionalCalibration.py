"""
Self-Supervised Backbone + Intersectional Calibration  (image, end-to-end fair pipeline)
자기지도 백본 + 교차집단 보정 (이미지용 공정 파이프라인)

MAF contribution — KAIST 유창동 연구실 (의료3 / ALINPneumothorax).

What this is (and is NOT)
-------------------------
This is the END-TO-END image pipeline behind algorithm ①
(IntersectionalThresholdCalibration), applied to a real medical-imaging setting:

    chest X-ray  ->  self-supervised backbone (RAD-DINO / DINOv2) embedding
                 ->  downstream classifier (Logistic Regression)
                 ->  INTERSECTIONAL (age x sex) equal-opportunity threshold
                     calibration  ->  fair decisions.

Important / honest scope:
- The self-supervised backbone is a PERFORMANCE / representation component
  (strong, label-free CXR features). It is NOT by itself a fairness method —
  a stronger model can even show a LARGER raw subgroup gap than a weak one.
- The FAIRNESS is provided by the intersectional post-hoc calibration (algorithm
  ①) applied on top of the backbone's scores. This module therefore reports
  fairness BEFORE vs AFTER calibration on the SAME (SSL-based) model, i.e. the
  gap that the calibration actually removes.

So the contribution demonstrated here = "a strong SSL backbone made fair, at the
intersectional (age x sex) level, by post-hoc calibration — on real chest X-rays."

Requirements
------------
torch, transformers, Pillow + chest-X-ray images and a manifest with columns
(uid,label,sex,age). See README for the data layout / download. CPU or GPU.

Run
---
  python algorithms/postprocessing/SSLBackbone_IntersectionalCalibration.py \
      --data_root /path/to/siim/iafoss512/train \
      --manifest  /path/to/siim/siim_manifest.csv  --n 1500
"""
import os
import sys
import argparse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

MIN_POS = 15          # min validation positives in a cell to fit its own threshold
AGE_EDGES = [50, 65]  # age bands: <50, 50-64, 65+


class SSLBackbone_Intersectional_Calibration:
    def __init__(self, data_root, manifest, n=1500,
                 backbone="microsoft/rad-dino", seed=1):
        self.data_root = data_root
        self.manifest = manifest
        self.n = n
        self.backbone = backbone
        self.seed = seed
        self.load_data()

    # --------------------------------------------------------------- data
    def load_data(self):
        df = pd.read_csv(self.manifest)
        cols = {c.lower(): c for c in df.columns}
        uid = cols.get('uid') or cols.get('imageid')
        lab, sex, age = cols.get('label'), (cols.get('sex') or cols.get('patientsex')), (cols.get('age') or cols.get('patientage'))
        df = df[[uid, lab, sex, age]].dropna()
        df.columns = ['uid', 'label', 'sex', 'age']
        df['label'] = df['label'].astype(int)
        df['sex'] = (df['sex'].astype(str).str.upper().str[0] == 'M').astype(int)   # M=1,F=0
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])
        df = df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        paths, rows = [], []
        for _, r in df.iterrows():
            p = os.path.join(self.data_root, str(r['uid']) + '.png')
            if os.path.exists(p):
                paths.append(p); rows.append(r)
            if len(paths) >= self.n:
                break
        self.meta = pd.DataFrame(rows).reset_index(drop=True)
        self.paths = paths
        self.meta['ageband'] = np.digitize(self.meta['age'].values, AGE_EDGES)
        print(f"[data] {len(paths)} imgs | pos={int(self.meta['label'].sum())} "
              f"| M={int((self.meta['sex']==1).sum())} F={int((self.meta['sex']==0).sum())}")

    # --------------------------------------------------- SSL backbone features
    def extract_ssl(self, batch_size=32):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        proc = AutoImageProcessor.from_pretrained(self.backbone)
        model = AutoModel.from_pretrained(self.backbone).to(device).eval()
        feats = []
        with torch.no_grad():
            for i in range(0, len(self.paths), batch_size):
                batch = [Image.open(p).convert('RGB') for p in self.paths[i:i + batch_size]]
                inp = proc(images=batch, return_tensors='pt').to(device)
                out = model(**inp)
                emb = out.pooler_output if getattr(out, 'pooler_output', None) is not None \
                    else out.last_hidden_state[:, 0]
                feats.append(emb.cpu().numpy())
        return np.concatenate(feats, axis=0)

    # --------------------------------------------------- fairness helpers
    @staticmethod
    def _tpr(y, p):
        pos = (y == 1)
        return float((p[pos] == 1).mean()) if pos.sum() else np.nan

    def _inter_deo(self, y, p, cells):
        tprs = []
        for c in set(cells):
            m = np.array([x == c for x in cells]) & (y == 1)
            if m.sum():
                tprs.append(float((p[m] == 1).mean()))
        return (max(tprs) - min(tprs)) if len(tprs) >= 2 else np.nan

    def _gender_deo(self, y, p, sex):
        return abs(self._tpr(y[sex == 0], p[sex == 0]) - self._tpr(y[sex == 1], p[sex == 1]))

    def _global_thr(self, sc, y):
        best_t, best_ba = 0.5, -1
        for t in np.linspace(0.01, 0.99, 99):
            p = (sc >= t).astype(int)
            ba = 0.5 * (self._tpr(y, p) + float((p[y == 0] == 0).mean() if (y == 0).sum() else np.nan))
            if ba > best_ba:
                best_ba, best_t = ba, t
        return best_t

    def _fit_cell_thr(self, sc, y, cells, gthr):
        target = self._tpr(y, (sc >= gthr).astype(int))
        thr = {}
        for c in set(cells):
            m = np.array([x == c for x in cells])
            if (m & (y == 1)).sum() < MIN_POS:
                thr[c] = gthr; continue
            best_t, best_gap = gthr, 1e9
            for t in np.linspace(0.01, 0.99, 99):
                p = (sc[m] >= t).astype(int)
                gap = abs(float((p[y[m] == 1] == 1).mean()) - target)
                if gap < best_gap:
                    best_gap, best_t = gap, t
            thr[c] = best_t
        return thr

    # ------------------------------------------------------------------ run
    def run(self):
        X = self.extract_ssl()
        y = self.meta['label'].values
        sex = self.meta['sex'].values
        cells = list(zip(self.meta['sex'].values, self.meta['ageband'].values))  # sex x ageband

        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(len(y))
        n_tr, n_va = int(0.6 * len(y)), int(0.2 * len(y))
        tr, va, te = idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=2000)
        lr.fit(scaler.fit_transform(X[tr]), y[tr])
        fav = np.where(lr.classes_ == 1)[0][0]
        sc_va = lr.predict_proba(scaler.transform(X[va]))[:, fav]
        sc_te = lr.predict_proba(scaler.transform(X[te]))[:, fav]

        gthr = self._global_thr(sc_va, y[va])
        cells_va = [cells[i] for i in va]; cells_te = [cells[i] for i in te]
        cthr = self._fit_cell_thr(sc_va, y[va], cells_va, gthr)

        # BEFORE: global threshold on SSL-model scores
        p_before = (sc_te >= gthr).astype(int)
        # AFTER: per-(sex x ageband) equal-opportunity thresholds
        p_after = np.array([1 if sc_te[i] >= cthr.get(cells_te[i], gthr) else 0 for i in range(len(te))])

        def report(p):
            return dict(accuracy=float((p == y[te]).mean()),
                        sensitivity=self._tpr(y[te], p),
                        gender_DEO=self._gender_deo(y[te], p, sex[te]),
                        intersectional_DEO=self._inter_deo(y[te], p, cells_te))
        return report(p_before), report(p_after)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--n', type=int, default=5000)  # >=~5000 recommended: smaller n overfits tiny cells
    ap.add_argument('--backbone', default='microsoft/rad-dino')
    a = ap.parse_args()

    algo = SSLBackbone_Intersectional_Calibration(a.data_root, a.manifest, n=a.n, backbone=a.backbone)
    before, after = algo.run()
    print("\nSSL backbone (RAD-DINO) + intersectional (sex x age) calibration")
    print(f"{'metric':24s}{'BEFORE(global)':>16s}{'AFTER(calibrated)':>18s}")
    for k in ["accuracy", "sensitivity", "gender_DEO", "intersectional_DEO"]:
        print(f"{k:24s}{before[k]:>16.4f}{after[k]:>18.4f}")
