"""
Build the SIIM-ACR manifest (uid,label,sex,age,view).

Data sources (public, Kaggle):
  - images + expert masks (512x512 PNG):
      https://www.kaggle.com/datasets/iafoss/siimacr-pneumothorax-segmentation-data-512
      -> <root>/iafoss512/train/<uid>.png , <root>/iafoss512/masks/<uid>.png
  - demographics parsed from the original DICOM headers:
      https://www.kaggle.com/datasets/safavieh/siim-acr-pneumothorax-segmentation-metadata
      train.csv -> <root>/meta_train.csv , test.csv -> <root>/meta_test.csv
      (columns: ImageId,ViewPosition,PatientAge,PatientSex)

Manifest fields:
  - uid    = ImageId = SIIM StudyInstanceUID (finest grouping available; the public
             release has NO separate PatientID, so splits are at this study level)
  - label  = 1 if the expert mask has ANY positive pixel (pneumothorax) else 0
  - sex    = PatientSex ('M'/'F' as released)
  - age    = PatientAge (int years)
  - view   = ViewPosition (PA/AP)

Usage:
  python tools/build_siim_manifest.py --root /path/to/siim --out siim_manifest.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image


def mask_label(mask_path):
    if not os.path.exists(mask_path):
        return None
    arr = np.asarray(Image.open(mask_path).convert('L'))
    return int((arr > 127).any())


def build_manifest(root, out):
    metas = [pd.read_csv(os.path.join(root, mf))
             for mf in ['meta_train.csv', 'meta_test.csv']
             if os.path.exists(os.path.join(root, mf))]
    if not metas:
        raise FileNotFoundError(f"meta_train.csv / meta_test.csv not found in {root}")
    meta = pd.concat(metas, ignore_index=True).drop_duplicates(subset='ImageId', keep='first')

    mask_dir = os.path.join(root, 'iafoss512', 'masks')
    rows = []
    for _, r in meta.iterrows():
        uid = str(r['ImageId'])
        lab = mask_label(os.path.join(mask_dir, uid + '.png'))
        if lab is None:
            continue
        rows.append(dict(uid=uid, label=lab,
                         sex=r['PatientSex'], age=r['PatientAge'], view=r['ViewPosition']))
    df = pd.DataFrame(rows).sort_values('uid').reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: n={len(df)} | pos={int(df['label'].sum())} "
          f"| M={(df['sex']=='M').sum()} F={(df['sex']=='F').sum()}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='SIIM root (contains iafoss512/, meta_*.csv)')
    ap.add_argument('--out', default='siim_manifest.csv')
    a = ap.parse_args()
    build_manifest(a.root, a.out)


if __name__ == '__main__':
    main()
