"""
Download the public data used by the algorithms.

  python download_data.py            # COMPAS + German + Adult -> AIF360 raw-data folder (① ② tabular)
  python download_data.py --siim     # + SIIM-ACR chest X-ray (~2.1 GB, Kaggle CLI + token)
                                     #   -> ./Sample/siim  (or $SIIM_ROOT)          (① image)

AIF360 does not ship COMPAS/German for licensing reasons; SIIM-ACR is fetched from Kaggle:
  images+masks : https://www.kaggle.com/datasets/iafoss/siimacr-pneumothorax-segmentation-data-512
  demographics : https://www.kaggle.com/datasets/safavieh/siim-acr-pneumothorax-segmentation-metadata
"""
import os
import sys
import urllib.request

import aif360

RAW = os.path.join(os.path.dirname(aif360.__file__), 'data', 'raw')
FILES = {
    'compas': ['https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'],
    'german': ['https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
               'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc'],
    'adult': ['https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
              'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'],
}

for name, urls in FILES.items():
    d = os.path.join(RAW, name)
    os.makedirs(d, exist_ok=True)
    for url in urls:
        dst = os.path.join(d, os.path.basename(url))
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f'[skip] {dst}')
            continue
        urllib.request.urlretrieve(url, dst)
        print(f'[ok]   {dst} ({os.path.getsize(dst)} bytes)')

if '--siim' in sys.argv:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from siim_dataset import SIIMDataset
    ds = SIIMDataset()
    if ds.is_downloaded():
        print(f'[skip] SIIM already present in {ds.ROOT}')
    else:
        ds.download()
