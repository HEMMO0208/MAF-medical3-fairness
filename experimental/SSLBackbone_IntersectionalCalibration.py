"""
Self-Supervised Backbone + Intersectional Calibration  (image, end-to-end fair pipeline)
자기지도 백본 + 교차집단 보정 (이미지용 공정 파이프라인)

MAF contribution — KAIST 유창동 연구실 (의료3 / ALINPneumothorax).

What this is (and is NOT)
-------------------------
End-to-end image pipeline behind algorithm ① (IntersectionalThresholdCalibration):

    chest X-ray -> self-supervised backbone (RAD-DINO/DINOv2) embedding
                -> downstream Logistic Regression -> INTERSECTIONAL (sex x age)
                   equal-opportunity threshold calibration -> fair decisions.

Honest scope:
- The self-supervised backbone is a PERFORMANCE / representation component, NOT by
  itself a fairness method. FAIRNESS is provided by the intersectional post-hoc
  calibration (algorithm ①) applied on top of the backbone's scores.
- This module reports fairness BEFORE vs AFTER calibration on the SAME (SSL-based)
  model. Calibration targets the INTERSECTIONAL (sex x age) gap; the single-axis
  gender gap can trade off in the other direction — both are reported.

Reproducibility
---------------
- rad-DINO pinned to revision REV (below).
- Split is by the manifest `uid` (= SIIM StudyInstanceUID, the finest grouping
  available in the public SIIM-ACR release; there is NO separate PatientID in the
  public metadata). Duplicate uids are dropped so no study appears in two splits.
  `--emit_split` writes the exact per-split uid lists + a SHA-256 of each.
- `--seeds` runs multiple split seeds (embeddings are extracted once and reused)
  and prints mean +/- std, plus per-(sex x age) positive counts and TPR.

Run
---
  python algorithms/postprocessing/SSLBackbone_IntersectionalCalibration.py \
      --data_root /path/to/siim/iafoss512/train \
      --manifest  /path/to/siim/siim_manifest.csv \
      --n 5000 --seeds 1 2 3 4 5 --emit_split split.json
"""
import os
import sys
import json
import hashlib
import argparse

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# --- reproducibility pins -------------------------------------------------
RAD_DINO_REVISION = "110cbc18d5133582e320b43d53bf5c44e410c936"  # microsoft/rad-dino
MIN_POS = 15            # min validation positives in a cell to fit its own threshold
AGE_EDGES = [50, 65]    # age bands: <50, 50-64, 65+
AGE_LABELS = ['<50', '50-64', '65+']
SEX_LABELS = {0: 'F', 1: 'M'}


def _parse_sex(s: pd.Series) -> np.ndarray:
    """Return 1=Male, 0=Female. Accepts strings ('M'/'F', 'male'/'female') OR
    numeric 0/1 (AIF360 convention: 1=privileged=Male). Raises if the column
    collapses to a single value (the silent-all-zero bug this guards against)."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(int).to_numpy()
        if not set(np.unique(out)).issubset({0, 1}):
            raise ValueError(f"numeric sex must be 0/1, got {np.unique(out)[:5]}")
    else:
        up = s.astype(str).str.strip().str.upper()
        out = up.str[0].map({'M': 1, 'F': 0}).to_numpy()
        if np.isnan(out.astype(float)).any():
            bad = up[~up.str[0].isin(['M', 'F'])].unique()[:5]
            raise ValueError(f"string sex must start with M/F, got {bad}")
        out = out.astype(int)
    if len(np.unique(out)) < 2:
        raise ValueError("sex column collapsed to a single value — check its format "
                         "(numeric 0/1 vs string M/F). See README.")
    return out


class SSLBackbone_Intersectional_Calibration:
    def __init__(self, data_root, manifest, n=5000,
                 backbone="microsoft/rad-dino", revision=RAD_DINO_REVISION, seed=1):
        self.data_root = data_root
        self.manifest = manifest
        self.n = n
        self.backbone = backbone
        self.revision = revision
        self.seed = seed
        self.load_data()

    # --------------------------------------------------------------- data
    def load_data(self):
        df = pd.read_csv(self.manifest)
        cols = {c.lower(): c for c in df.columns}
        uid = cols.get('uid') or cols.get('imageid')
        lab = cols.get('label')
        sex = cols.get('sex') or cols.get('patientsex')
        age = cols.get('age') or cols.get('patientage')
        df = df[[uid, lab, sex, age]].dropna()
        df.columns = ['uid', 'label', 'sex', 'age']
        df = df.drop_duplicates(subset='uid', keep='first')          # study-level uniqueness
        df['label'] = df['label'].astype(int)
        df['sex'] = _parse_sex(df['sex'])                            # robust M/F or 0/1
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['age'])
        # deterministic subset (fixed order by uid, then take first n present on disk)
        df = df.sort_values('uid').reset_index(drop=True)

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
        assert self.meta['uid'].is_unique, "duplicate uid after dedup — split would leak"
        print(f"[data] {len(paths)} unique-uid imgs | pos={int(self.meta['label'].sum())} "
              f"| M={int((self.meta['sex']==1).sum())} F={int((self.meta['sex']==0).sum())}")

    # --------------------------------------------------- SSL backbone features
    def extract_ssl(self, batch_size=32):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        proc = AutoImageProcessor.from_pretrained(self.backbone, revision=self.revision)
        model = AutoModel.from_pretrained(self.backbone, revision=self.revision).to(device).eval()
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
        tprs = [float((p[(np.array([x == c for x in cells])) & (y == 1)] == 1).mean())
                for c in set(cells) if ((np.array([x == c for x in cells])) & (y == 1)).sum()]
        return (max(tprs) - min(tprs)) if len(tprs) >= 2 else np.nan

    def _gender_deo(self, y, p, sex):
        return abs(self._tpr(y[sex == 0], p[sex == 0]) - self._tpr(y[sex == 1], p[sex == 1]))

    def _global_thr(self, sc, y):
        best_t, best_ba = 0.5, -1
        for t in np.linspace(0.01, 0.99, 99):
            p = (sc >= t).astype(int)
            tnr = float((p[y == 0] == 0).mean()) if (y == 0).sum() else np.nan
            ba = 0.5 * (self._tpr(y, p) + tnr)
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

    def _split(self, seed):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.paths))
        n_tr, n_va = int(0.6 * len(idx)), int(0.2 * len(idx))
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]

    # ------------------------------------------------------------- one run
    def _one(self, X, y, sex, cells, seed):
        tr, va, te = self._split(seed)
        scaler = StandardScaler(); lr = LogisticRegression(max_iter=2000)
        lr.fit(scaler.fit_transform(X[tr]), y[tr])
        fav = np.where(lr.classes_ == 1)[0][0]
        sc_va = lr.predict_proba(scaler.transform(X[va]))[:, fav]
        sc_te = lr.predict_proba(scaler.transform(X[te]))[:, fav]
        gthr = self._global_thr(sc_va, y[va])
        cva = [cells[i] for i in va]; cte = [cells[i] for i in te]
        cthr = self._fit_cell_thr(sc_va, y[va], cva, gthr)
        p_before = (sc_te >= gthr).astype(int)
        p_after = np.array([1 if sc_te[i] >= cthr.get(cte[i], gthr) else 0 for i in range(len(te))])

        def rep(p):
            return dict(accuracy=float((p == y[te]).mean()), sensitivity=self._tpr(y[te], p),
                        gender_DEO=self._gender_deo(y[te], p, sex[te]),
                        intersectional_DEO=self._inter_deo(y[te], p, cte))
        # per-cell positive counts + TPR (test), before/after
        percell = {}
        for c in sorted(set(cte)):
            m = np.array([x == c for x in cte]); pos = m & (y[te] == 1)
            name = f"{SEX_LABELS[c[0]]}x{AGE_LABELS[c[1]]}"
            percell[name] = dict(n_pos=int(pos.sum()),
                                 TPR_before=float((p_before[pos] == 1).mean()) if pos.sum() else None,
                                 TPR_after=float((p_after[pos] == 1).mean()) if pos.sum() else None)
        return rep(p_before), rep(p_after), percell, (tr, va, te)

    def run(self, seeds=(1,), emit_split=None):
        X = self.extract_ssl()
        y = self.meta['label'].values; sex = self.meta['sex'].values
        cells = list(zip(self.meta['sex'].values, self.meta['ageband'].values))
        agg = {'before': [], 'after': []}
        first = None
        for s in seeds:
            before, after, percell, split = self._one(X, y, sex, cells, s)
            agg['before'].append(before); agg['after'].append(after)
            print(f"[seed {s}] interDEO {before['intersectional_DEO']:.3f}->{after['intersectional_DEO']:.3f} "
                  f"| genderDEO {before['gender_DEO']:.3f}->{after['gender_DEO']:.3f} "
                  f"| acc {before['accuracy']:.3f}->{after['accuracy']:.3f}")
            if first is None:
                first = (percell, split)
        # summary
        def ms(key, which):
            v = np.array([d[key] for d in agg[which]])
            return float(np.nanmean(v)), float(np.nanstd(v))
        print(f"\n=== {len(seeds)} seeds: mean +/- std ===")
        for key in ['accuracy', 'sensitivity', 'gender_DEO', 'intersectional_DEO']:
            mb, sb = ms(key, 'before'); ma, sa = ms(key, 'after')
            print(f"{key:20s} before {mb:.4f}+/-{sb:.4f}   after {ma:.4f}+/-{sa:.4f}")
        print("\n=== per-(sex x age) cell: positives & TPR (seed %d) ===" % seeds[0])
        print(json.dumps(first[0], indent=2, ensure_ascii=False))

        if emit_split:
            tr, va, te = first[1]
            uids = self.meta['uid'].values
            def h(ix): return hashlib.sha256((",".join(sorted(uids[ix]))).encode()).hexdigest()
            rec = {'seed': seeds[0], 'n': len(self.paths),
                   'rad_dino_revision': self.revision,
                   'counts': {'train': len(tr), 'valid': len(va), 'test': len(te)},
                   'sha256': {'train': h(tr), 'valid': h(va), 'test': h(te)},
                   'uids': {'train': sorted(uids[tr].tolist()),
                            'valid': sorted(uids[va].tolist()),
                            'test': sorted(uids[te].tolist())}}
            with open(emit_split, 'w') as f:
                json.dump(rec, f, indent=2)
            print(f"\n[split] written to {emit_split} | sha256(test)={rec['sha256']['test'][:16]}...")
        return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--n', type=int, default=5000)  # >=~5000: smaller n overfits tiny cells
    ap.add_argument('--backbone', default='microsoft/rad-dino')
    ap.add_argument('--revision', default=RAD_DINO_REVISION)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--emit_split', default=None)
    a = ap.parse_args()
    algo = SSLBackbone_Intersectional_Calibration(a.data_root, a.manifest, n=a.n,
                                                  backbone=a.backbone, revision=a.revision)
    algo.run(seeds=tuple(a.seeds), emit_split=a.emit_split)
