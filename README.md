# MAF 추가 알고리즘 — KAIST 유창동 연구실 (의료3 / ALINPneumothorax)

MSIT AI FAIR (MAF) 프레임워크에 추가할 **공정성(Fairness) 알고리즘** 2종.
AIF360 기반 MAF2024_guide 양식(aifData + LogisticRegression baseline +
`metrics.common_utils`)에 맞춰 작성되었으며, 정형 벤치마크(COMPAS/German/Adult)에서
바로 실행·검증됩니다. 두 알고리즘 모두 **교차집단(intersectional, 예: 성별×인종)**
공정성을 다루는 것이 특징이며, 모델·도메인 불가지론이라 이미지 모델의 예측 점수에도
동일하게 적용됩니다(원 연구: 흉부 X선 기흉 탐지).

## 쉬운 설명 (처음 보는 분을 위해)

의료 AI가 "병이 있다/없다"를 잘 맞히더라도, **특정 집단(예: 고령 남성, 특정 성별·인종)에게만
유독 자주 틀리면** 공정하지 않습니다. 아래 알고리즘들은 그 집단 간 격차를 줄여줍니다.
특히 "성별만", "나이만" 따로 보면 괜찮아 보여도 **성별×나이처럼 둘을 겹쳐 보면 드러나는
사각지대(교차집단, intersectional)** 를 다루는 것이 공통된 특징입니다.

- **① 교차집단 사후 임계값 보정** — AI는 보통 "점수가 0.5를 넘으면 양성"이라는 **하나의
  기준선을 모두에게 똑같이** 적용합니다. 그러면 어떤 집단은 유독 많이 놓칩니다. 이 방법은
  **집단마다 기준선을 조금씩 다르게** 맞춰, 모든 집단이 비슷한 비율로 병을 잡아내게 합니다.
  **모델을 다시 학습할 필요 없이** 이미 학습된 모델의 판정 기준만 바꿉니다. (예측 '이후' 단계)

- **② 교차집단 균형 리샘플링** — AI는 학습 때 **많이 본 집단은 잘, 적게 본 집단은 못** 맞힙니다.
  데이터에 적게 들어있는 집단의 사례를 **복제해 더 자주 보여주어** 모든 집단을 골고루 학습시킵니다.
  (학습 '이전' 데이터 단계)

- **③ 자기지도 백본 + 교차집단 보정** — 먼저 **정답 라벨 없이 대량의 X선 사진만으로 스스로
  영상 특징을 익힌 모델(자기지도 학습)** 을 쓰면 적은 정답으로도 성능이 크게 좋아집니다.
  그 위에 ①의 집단별 기준선 보정을 얹어 **성능과 공정성을 함께** 잡는 이미지용 파이프라인입니다.
  (자기지도 학습은 '성능'을 위한 것이고, '공정성'은 ①이 담당합니다.)

---

## 포함 알고리즘

| # | 범주 | 파일 | 요약 |
|---|------|------|------|
| ① | post-processing | `algorithms/postprocessing/IntersectionalThresholdCalibration.py` | 학습된 분류기의 점수에 대해 **교차집단별 등가기회(equal-opportunity) 임계값**을 조정, 재학습 없이 집단 간 TPR 격차 보정. 희소 셀은 전역 임계값으로 폴백. |
| ② | pre-processing | `algorithms/preprocessing/SubgroupBalancedResampling.py` | (라벨×교차집단) 그룹을 **빈도 역가중 오버샘플링**하여 학습 데이터의 집단 불균형 편향을 완화. |
| ③ | representation + post | `algorithms/postprocessing/SSLBackbone_IntersectionalCalibration.py` | **자기지도 백본(RAD-DINO/DINOv2) 임베딩 + ①의 교차집단 보정**을 결합한 이미지용 엔드투엔드 공정 파이프라인(흉부 X선). SSL은 성능 요소, 공정성은 ①이 담당. |

새로 도입한 지표: **intersectional-DEO** = (교차집단 TPR의 최대 − 최소). 낮을수록 공정.
AIF360 표준 지표(`equal_opportunity_difference`, `average_abs_odds_difference` 등)는
주(primary) 보호속성 기준으로 함께 보고됩니다.

## 실행 환경
- Python 3.8–3.10
- 설치: `pip install -r requirements.txt`  (aif360, scikit-learn, numpy, pandas)

## 데이터셋 준비 (AIF360 벤치마크)
AIF360은 라이선스상 원본 데이터를 동봉하지 않으므로 아래를 받아 지정 폴더에 두세요
(`<AIF360>` = `python -c "import aif360,os;print(os.path.dirname(aif360.__file__))"`).

- **COMPAS** (기본 데모):
  `https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv`
  → `<AIF360>/data/raw/compas/compas-scores-two-years.csv`
- **German** (선택):
  `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data` (+ `german.doc`)
  → `<AIF360>/data/raw/german/`

## 실행
```bash
# ① 교차집단 사후 임계값 보정
python algorithms/postprocessing/IntersectionalThresholdCalibration.py
# ② 교차집단 균형 리샘플링
python algorithms/preprocessing/SubgroupBalancedResampling.py
```
각 스크립트는 **보정 전/후** 지표를 출력합니다.

## 검증된 예시 출력 (COMPAS, 성별×인종 4개 교차셀)
① IntersectionalThresholdCalibration
```
metric                            BEFORE(global)   AFTER(per-cell)
accuracy                                  0.6733            0.6335
equal_opportunity_difference             -0.1840            0.0644
average_abs_odds_difference               0.2080            0.0473
intersectional_DEO                        0.4142            0.1859
```
② SubgroupBalancedResampling
```
metric                            BEFORE(original)   AFTER(balanced)
equal_opportunity_difference               -0.1263           -0.1060
average_abs_odds_difference                 0.1873            0.1265
intersectional_DEO                          0.2564            0.1718
```
(정확도는 소폭 하락 — 공정성 개선에 따른 통상적 트레이드오프.)

③ SSLBackbone_IntersectionalCalibration — 흉부 X선 SIIM-ACR 5,000장, rad-DINO 임베딩,
교차집단 = 성별×연령(3밴드), 검증값:
```
metric                    BEFORE(global)  AFTER(calibrated)
accuracy                          0.7660            0.7650
sensitivity                       0.7675            0.7632
gender_DEO                        0.0237            0.0852
intersectional_DEO                0.2371            0.1954
```
- **교차집단-DEO가 보정 후 감소(0.237→0.195), 정확도 유지** — 알고리즘 목표(교차 등가기회) 달성.
- ⚠️ 정직한 주의: (a) **자기지도 백본 자체는 공정성 기법이 아님**(성능/표현 요소). 공정성은 ①의 교차집단 보정이 담당. (b) 교차 등가기회를 맞추는 과정에서 **단일축 성별-DEO는 트레이드오프로 증가**할 수 있음. (c) 교차셀이 충분해야 일반화됨 — **n이 작으면(예: 1,500) 셀 과적합으로 개선이 사라짐**(실측 확인). n≥~5,000 권장.
- 실행: `python algorithms/postprocessing/SSLBackbone_IntersectionalCalibration.py --data_root <X선폴더> --manifest <uid,label,sex,age csv> --n 5000` (torch·transformers 필요, rad-DINO는 HF에서 자동 다운로드)

## MAF 통합 방법
`algorithms/`·`metrics/` 폴더를 MAF2024_guide 트리에 그대로 병합하면 됩니다
(패키지 구조·import 규약이 가이드와 동일). 다른 데이터셋은 각 스크립트의
`dataset_name`·`protected_attrs` 인자로 교체하거나, 가이드의 `aifData`로 변환해
`(scores, labels, protected_attributes)`만 전달하면 적용됩니다.

## 비정형(이미지) 적용 메모
두 알고리즘은 **예측 점수·보호속성·라벨**만 사용하므로, 이미지 모델의 출력 점수에 그대로
적용됩니다. 원 연구에서는 흉부 X선 기흉 탐지(SIIM-ACR, 성별×연령 교차집단)에서 ①을 적용해
교차집단 격차를 크게 낮추었습니다. 이미지 baseline을 가이드 방식(픽셀 평탄화 + LR)으로
구성하려면 `MAF2024_guide/sample.py`의 `aifData` 변환을 참고하세요.

## 출처 / 연구실
KAIST 유창동 연구실 · AI-DEP 의료3(ALINPneumothorax). 문의 시 연구실로 연락.

