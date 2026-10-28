"""
SIIM-ACR Pneumothorax chest X-ray dataset for MAF, in the same form as the guide's
PubFigDataset (MSIT AI FAIR guideline p.22-26):

  - download()   : fetch the public data (Kaggle CLI)
  - to_dataset() : load images, resize to 64x64, flatten pixels into a DataFrame,
                   attach target + protected attributes, convert to AIF360 `aifData`

Returned dict (same keys as the guide):
  {'aif_dataset', 'image_list', 'attribute', 'target', 'bias'}

Columns in the aifData
  - pixel features : 64*64 grayscale values scaled to [0,1]
  - target         : 'Pneumothorax' (1 = present, expert mask non-empty; favorable = 1)
  - protected      : 'sex'          (1 = Male,      0 = Female)      privileged = 1
                     'age_under65'  (1 = age < 65,  0 = age >= 65)   privileged = 1
  Two protected attributes are kept so that intersectional (sex x age) cells can be
  formed; AIF360 single-attribute metrics use 'sex' as the primary attribute.

Data layout under ROOT (default ./Sample/siim):
  iafoss512/train/<uid>.png   iafoss512/masks/<uid>.png   meta_train.csv   meta_test.csv
"""
import os
import sys
import shutil
import subprocess

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from aif360.datasets import StandardDataset

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.build_siim_manifest import build_manifest


class aifData(StandardDataset):
    def __init__(self, df, label_name, favorable_classes,
                 protected_attribute_names, privileged_classes,
                 instance_weights_name='', scores_name='',
                 categorical_features=[], features_to_keep=[],
                 features_to_drop=[], na_values=[], custom_preprocessing=None,
                 metadata=None):
        super(aifData, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)


class SIIMDataset:
    IMAGE_DATASET = 'iafoss/siimacr-pneumothorax-segmentation-data-512'
    META_DATASET = 'safavieh/siim-acr-pneumothorax-segmentation-metadata'
    TARGET_NAME = 'Pneumothorax'
    PROTECTED = ['sex', 'age_under65']

    RAD_DINO = 'microsoft/rad-dino'
    RAD_DINO_REVISION = '110cbc18d5133582e320b43d53bf5c44e410c936'

    def __init__(self, root=None, img_size=64, feature='pixel'):
        """feature='pixel'   : 64x64 grayscale pixels flattened (guide p.25, literal)
           feature='raddino' : frozen RAD-DINO embedding (768-d) per image, used as the
                               aifData feature vector; the LR baseline is unchanged."""
        if feature not in ('pixel', 'raddino'):
            raise ValueError(feature)
        self.ROOT = root or os.environ.get('SIIM_ROOT', './Sample/siim')
        self.img_size = img_size
        self.feature = feature
        self.MANIFEST = os.path.join(self.ROOT, 'siim_manifest.csv')
        self.CACHE = os.path.join(self.ROOT, f'siim{img_size}_cache.npz')
        self.EMB_CACHE = os.path.join(self.ROOT, 'siim_raddino_cache.npz')

    def is_downloaded(self):
        return (os.path.isdir(os.path.join(self.ROOT, 'iafoss512', 'train'))
                and os.path.exists(os.path.join(self.ROOT, 'meta_train.csv')))

    def download(self):
        """Requires the Kaggle CLI and an API token (~/.kaggle/kaggle.json or access_token)."""
        os.makedirs(self.ROOT, exist_ok=True)
        img_dir = os.path.join(self.ROOT, 'iafoss512')
        print(f'Downloading {self.IMAGE_DATASET} (~2.1 GB) -> {img_dir}')
        subprocess.run(['kaggle', 'datasets', 'download', '-d', self.IMAGE_DATASET,
                        '-p', img_dir, '--unzip'], check=True)
        print(f'Downloading {self.META_DATASET} -> {self.ROOT}')
        subprocess.run(['kaggle', 'datasets', 'download', '-d', self.META_DATASET,
                        '-p', self.ROOT, '--unzip'], check=True)
        for src, dst in [('train.csv', 'meta_train.csv'), ('test.csv', 'meta_test.csv')]:
            s, d = os.path.join(self.ROOT, src), os.path.join(self.ROOT, dst)
            if os.path.exists(s):
                shutil.move(s, d)
        print('Download finished.')

    def _load_pixels(self, manifest):
        if os.path.exists(self.CACHE):
            z = np.load(self.CACHE, allow_pickle=True)
            if list(z['uids']) == list(manifest['uid']):
                return list(z['images'])
        img_dir = os.path.join(self.ROOT, 'iafoss512', 'train')
        images = []
        for uid in tqdm(manifest['uid'], desc='load+resize'):
            im = Image.open(os.path.join(img_dir, uid + '.png')).convert('L')
            images.append(np.asarray(im.resize((self.img_size, self.img_size)), dtype=np.uint8))
        np.savez_compressed(self.CACHE, uids=manifest['uid'].values, images=np.stack(images))
        return images

    def _load_raddino(self, manifest, batch_size=32):
        if os.path.exists(self.EMB_CACHE):
            z = np.load(self.EMB_CACHE, allow_pickle=True)
            if list(z['uids']) == list(manifest['uid']) and str(z['revision']) == self.RAD_DINO_REVISION:
                return z['emb']
        import torch
        from transformers import AutoModel, AutoImageProcessor
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        proc = AutoImageProcessor.from_pretrained(self.RAD_DINO, revision=self.RAD_DINO_REVISION)
        model = AutoModel.from_pretrained(self.RAD_DINO, revision=self.RAD_DINO_REVISION).to(device).eval()
        img_dir = os.path.join(self.ROOT, 'iafoss512', 'train')
        uids = list(manifest['uid'])
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0, len(uids), batch_size), desc='rad-dino'):
                batch = [Image.open(os.path.join(img_dir, u + '.png')).convert('RGB')
                         for u in uids[i:i + batch_size]]
                out = model(**proc(images=batch, return_tensors='pt').to(device))
                emb = out.pooler_output if getattr(out, 'pooler_output', None) is not None \
                    else out.last_hidden_state[:, 0]
                feats.append(emb.cpu().numpy())
        emb = np.concatenate(feats).astype(np.float32)
        np.savez_compressed(self.EMB_CACHE, uids=np.array(uids), emb=emb,
                            revision=self.RAD_DINO_REVISION)
        return emb

    def to_dataset(self):
        if not os.path.exists(self.MANIFEST):
            build_manifest(self.ROOT, self.MANIFEST)
        man = pd.read_csv(self.MANIFEST)
        img_dir = os.path.join(self.ROOT, 'iafoss512', 'train')
        man = man[man['uid'].apply(lambda u: os.path.exists(os.path.join(img_dir, u + '.png')))]
        man = man.dropna(subset=['sex', 'age']).drop_duplicates(subset='uid')
        man = man.sort_values('uid').reset_index(drop=True)
        assert man['uid'].is_unique

        sex = man['sex'].astype(str).str.strip().str.upper().str[0]
        if not set(sex.unique()) <= {'M', 'F'}:
            raise ValueError(f"manifest 'sex' must be 'M'/'F', got {sorted(sex.unique())[:5]}")
        man['sex_bin'] = (sex == 'M').astype(int)
        man['age_under65'] = (pd.to_numeric(man['age']) < 65).astype(int)

        img_list = self._load_pixels(man)
        if self.feature == 'pixel':
            X = np.stack([im.ravel() for im in img_list]).astype(np.float32) / 255.0
            prefix = 'px'
        else:
            X = self._load_raddino(man)
            prefix = 'emb'
        df = pd.DataFrame(X)
        df.columns = [f'{prefix}{i}' for i in range(X.shape[1])]
        df[self.TARGET_NAME] = man['label'].astype(int).values
        df['sex'] = man['sex_bin'].values
        df['age_under65'] = man['age_under65'].values
        df.index = man['uid'].values

        dataset = aifData(df=df, label_name=self.TARGET_NAME,
                          favorable_classes=[1],
                          protected_attribute_names=self.PROTECTED,
                          privileged_classes=[[1], [1]])
        print(f"[SIIM:{self.feature}] n={len(df)} | dim={X.shape[1]} | pos={int(df[self.TARGET_NAME].sum())} "
              f"| M={int(df['sex'].sum())} F={int((df['sex']==0).sum())} "
              f"| age<65={int(df['age_under65'].sum())} age>=65={int((df['age_under65']==0).sum())}")
        return {'aif_dataset': dataset,
                'image_list': img_list,
                'attribute': man,
                'target': df[self.TARGET_NAME].values,
                'bias': df['sex'].values}
