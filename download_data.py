"""
Download the public benchmark files required by algorithms ① ② into AIF360's
raw-data folder (AIF360 does not ship them for licensing reasons).

  python download_data.py            # COMPAS + German
"""
import os
import urllib.request

import aif360

RAW = os.path.join(os.path.dirname(aif360.__file__), 'data', 'raw')
FILES = {
    'compas': ['https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv'],
    'german': ['https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
               'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.doc'],
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
