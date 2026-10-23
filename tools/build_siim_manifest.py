"""
Build the SIIM-ACR manifest (uid,label,sex,age,view) used by algorithm ③.

Data source (SIIM-ACR Pneumothorax Segmentation, public):
  - images : <root>/iafoss512/train/<uid>.png        (512x512 PNG, 1 per StudyInstanceUID)
  - masks  : <root>/iafoss512/masks/<uid>.png         (expert pixel mask; empty = no pneumothorax)
  - meta   : <root>/meta_train.csv , meta_test.csv    (columns: ImageId,ViewPosition,PatientAge,PatientSex)
             (demographics parsed from the original DICOM headers)

Manifest fields:
  - uid    = ImageId = SIIM StudyInstanceUID  (finest grouping available; NO separate
             PatientID exists in the public release -> splits are at this study level)
  - label  = 1 if the expert mask has ANY positive pixel (pneumothorax present) else 0
  - sex    = PatientSex   (string 'M'/'F' as released)
  - age    = PatientAge   (int years)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='SIIM root (contains iafoss512/, meta_*.csv)')
    ap.add_argument('--out', default='siim_manifest.csv')
    a = ap.parse_args()

    metas = []
    for mf in ['meta_train.csv', 'meta_test.csv']:
        p = os.path.join(a.root, mf)
        if os.path.exists(p):
            metas.append(pd.read_csv(p))
    meta = pd.concat(metas, ignore_index=True).drop_duplicates(subset='ImageId', keep='first')

    mask_dir = os.path.join(a.root, 'iafoss512', 'masks')
    rows = []
    for _, r in meta.iterrows():
        uid = str(r['ImageId'])
        lab = mask_label(os.path.join(mask_dir, uid + '.png'))
        if lab is None:
            continue
        rows.append(dict(uid=uid, label=lab,
                         sex=r['PatientSex'], age=r['PatientAge'], view=r['ViewPosition']))
    df = pd.DataFrame(rows).sort_values('uid').reset_index(drop=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}: n={len(df)} | pos={int(df['label'].sum())} "
          f"| M={(df['sex']=='M').sum()} F={(df['sex']=='F').sum()}")


if __name__ == '__main__':
    main()
