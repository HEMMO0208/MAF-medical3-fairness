"""
Intersectional Equalized-Opportunity Threshold Calibration — IMAGE version
교차집단 등가기회 임계값 보정 (이미지, 흉부 X선)

Algorithm ① applied to chest X-rays, following the MAF guideline's image pipeline
(MSIT AI FAIR guideline p.22-30, reference: FairFeatureDistillation.py):

  1. SIIMDataset.to_dataset(): image -> feature vector -> DataFrame -> AIF360 aifData
     (target = Pneumothorax, protected = sex, age_under65)
     feature='raddino' (default): frozen RAD-DINO embedding (768-d)
     feature='pixel'            : 64x64 grayscale pixels flattened (the guide's literal example)
  2. split with aifData.split(..., shuffle=True, seed=1)  (default 50/25/25)
  3. baseline_fit(): LogisticRegression on those features
  4. fairness: per-(sex x age) equal-opportunity thresholds fit on validation,
     applied to test -> transformed test predictions as an aifData
  5. compute_metrics(): metrics.common_utils.compute_metrics (AIF360), plus
     intersectional_DEO (max-min TPR over sex x age cells)
  6. run() -> metrics_orig, metrics_transf

The calibration logic is inherited unchanged from the tabular algorithm
(IntersectionalThresholdCalibration.py); only data loading differs.

Result (5 seeds, see README): the fairness improvement does NOT reproduce on SIIM-ACR —
with a good baseline the intersectional gap before calibration is already at chance
level, so there is nothing measurable to correct. Provided as ①'s image adapter.

Run
---
  python download_data.py --siim          # or point SIIM_ROOT to an existing copy
  python algorithms/postprocessing/IntersectionalThresholdCalibration_Image.py
"""
import os
import sys

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '../../'))
sys.path.append(parent_dir)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from siim_dataset import SIIMDataset
from algorithms.postprocessing.IntersectionalThresholdCalibration import Intersectional_Threshold_Calibration

np.random.seed(1)


class Intersectional_Threshold_Calibration_Image(Intersectional_Threshold_Calibration):
    def __init__(self, dataset_name: str = 'siim', protected: str = 'sex',
                 protected_attrs=None, image_shape=(1, 64, 64), seed: int = 1,
                 data_root=None, feature: str = 'raddino', train_frac: float = 0.5,
                 **kwargs) -> None:
        """feature   : 'raddino' (frozen RAD-DINO embedding, default) or 'pixel'
                       (64x64 flattened pixels, the guide's literal example).
           train_frac: fraction for training; the remainder is split 50/50 into
                       validation (threshold fitting) and test. Default 0.5 -> 50/25/25."""
        self.dataset_name = dataset_name
        self.protected = protected
        self.primary = protected
        self.protected_attrs = protected_attrs or ['sex', 'age_under65']
        self.image_shape = image_shape
        self.seed = seed
        self.data_root = data_root
        self.feature = feature
        self.train_frac = train_frac
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        if self.dataset_name == 'siim':
            self.siim = SIIMDataset(root=self.data_root, img_size=self.image_shape[-1],
                                    feature=self.feature)
            if not self.siim.is_downloaded():
                self.siim.download()
            self.dataset = self.siim.to_dataset()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.dataset_orig = self.dataset['aif_dataset']
        self.privileged_groups = [{self.protected: 1}]
        self.unprivileged_groups = [{self.protected: 0}]

        # train / valid (threshold fitting) / test ; default 50 / 25 / 25
        self.dataset_train, dataset_vt = self.dataset_orig.split([self.train_frac], shuffle=True, seed=self.seed)
        self.dataset_valid, self.dataset_test = dataset_vt.split([0.5], shuffle=True, seed=self.seed)

        self.fav = self.dataset_orig.favorable_label
        self.unfav = self.dataset_orig.unfavorable_label
        self.attr_idx = {n: i for i, n in enumerate(self.dataset_orig.protected_attribute_names)}

    def baseline_fit(self):
        """LogisticRegression on the image features (guide p.29). Returns valid/test
        predictions carrying scores, which the calibration step thresholds."""
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(self.dataset_train.features)
        y_tr = self.dataset_train.labels.ravel()
        lmod = LogisticRegression(max_iter=1000, random_state=1)
        lmod.fit(X_tr, y_tr)
        fav_col = np.where(lmod.classes_ == self.fav)[0][0]

        valid_pred = self.dataset_valid.copy(deepcopy=True)
        test_pred = self.dataset_test.copy(deepcopy=True)
        valid_pred.scores = lmod.predict_proba(scaler.transform(self.dataset_valid.features))[:, fav_col].reshape(-1, 1)
        test_pred.scores = lmod.predict_proba(scaler.transform(self.dataset_test.features))[:, fav_col].reshape(-1, 1)
        return valid_pred, test_pred


if __name__ == "__main__":
    clf = Intersectional_Threshold_Calibration_Image(dataset_name='siim', protected='sex')
    m_orig, m_transf = clf.run()
    keys = ["accuracy", "Balanced accuracy", "true_positive_rate",
            "equal_opportunity_difference", "average_abs_odds_difference", "intersectional_DEO"]
    print(f"\n{'metric':32s}{'BEFORE(global)':>16s}{'AFTER(per-cell)':>18s}")
    for k in keys:
        print(f"{k:32s}{m_orig[k]:>16.4f}{m_transf[k]:>18.4f}")
