# MAF 추가 알고리즘 — KAIST 유창동 연구실 (AI-DEP 의료3 / ALINPneumothorax)

MSIT AI FAIR(MAF) 프레임워크에 추가할 **공정성(Fairness) 알고리즘**입니다.
첨부받은 MAF2024_guide 양식(`aifData` + Logistic Regression baseline + `metrics.common_utils`)을 따릅니다.

> **최종 제출 알고리즘: 2종** — ① 교차집단 사후 임계값 보정, ② 교차집단 균형 리샘플링.
> ①은 정형 데이터용과 **이미지 데이터 연동 모듈**(가이드 22~30쪽 이미지 처리 방식)을 함께 제공합니다.
> 협약 성과지표(정확도 94 / 공정성 5 / 정책하락률 3) 재현 코드는 별도 폴더 `performance_targets/`에 있습니다.

| 폴더/파일 | 내용 |
|---|---|
| `algorithms/postprocessing/IntersectionalThresholdCalibration.py` | ① 정형 데이터 |
| `algorithms/postprocessing/IntersectionalThresholdCalibration_Image.py` | ① 이미지 연동 (SIIM-ACR 흉부 X선) |
| `algorithms/preprocessing/SubgroupBalancedResampling.py` | ② 정형 데이터 |
| `siim_dataset.py` | SIIM-ACR 데이터셋 클래스 (가이드 `PubFigDataset`과 같은 `download()`/`to_dataset()` 구조) |
| `metrics/common_utils.py` | AIF360 지표 (MAF2024_guide 재사용) |
| `download_data.py` | 데이터 자동 다운로드 |
| `tools/multiseed.py`, `results/` | 시드 반복 검증 스크립트와 결과(JSON·로그) |
| `tools/build_siim_manifest.py` | SIIM 라벨·성별·나이 표 생성 |
| `performance_targets/` | 협약 성과지표 재현 코드·결과 |
| `experimental/` | 미제출 실험(자기지도 백본 + 보정) — 재현되지 않아 철회 |

---

## 쉬운 설명 (처음 보는 분을 위해)

AI가 "양성/음성"을 평균적으로 잘 맞혀도, **특정 집단(예: 특정 성별×인종, 고령 남성)에게만 자주 틀리면** 공정하지 않습니다.
특히 "성별만", "인종만" 따로 보면 괜찮아 보여도 **둘을 겹쳐 보면 드러나는 사각지대(교차집단)** 가 있습니다.

- **① 교차집단 사후 임계값 보정** — AI는 보통 "점수 0.5 이상이면 양성"이라는 **하나의 기준선을 모두에게 똑같이** 적용합니다.
  그러면 어떤 집단은 유독 많이 놓칩니다. 이 방법은 **집단마다 기준선을 조금씩 다르게** 맞춰, 모든 집단이 비슷한 비율로 양성을
  잡게 합니다. **모델을 다시 학습하지 않고** 판정 기준만 바꿉니다. (예측 '이후' 단계)
- **② 교차집단 균형 리샘플링** — AI는 학습 때 **많이 본 집단은 잘, 적게 본 집단은 못** 맞힙니다. 데이터에 적은 집단의 사례를
  **복제해 더 자주 보여주어** 골고루 학습시킵니다. (학습 '이전' 단계)

새로 쓰는 지표 **intersectional-DEO** = 교차집단별 TPR(실제 양성을 맞힌 비율) 중 최대 − 최소. 낮을수록 공정합니다.
AIF360 표준 지표(`equal_opportunity_difference`, `average_abs_odds_difference` 등)도 주 보호속성 기준으로 함께 출력합니다.

---

## 빠른 실행 (Quick start)
```bash
pip install -r requirements.txt
python download_data.py                                                  # COMPAS · German · Adult 원본을 AIF360 폴더로
python algorithms/postprocessing/IntersectionalThresholdCalibration.py   # ① (COMPAS)
python algorithms/preprocessing/SubgroupBalancedResampling.py            # ② (COMPAS)
```
각 스크립트는 보정 전/후 지표를 출력합니다. 데이터셋은 클래스 인자 `dataset_name='compas'|'adult'|'german'`으로 바꿉니다.

이미지 연동 모듈:
```bash
python download_data.py --siim      # SIIM-ACR (~2.1GB, Kaggle CLI 및 API 토큰 필요) -> ./Sample/siim
python algorithms/postprocessing/IntersectionalThresholdCalibration_Image.py
```
이미 SIIM 데이터가 있으면 `SIIM_ROOT=<경로>`로 지정합니다.

---

## 검증 결과 (5개 split seed, 평균 ± 표준편차)

### 정형 데이터 — 적용 가능 데이터셋: **COMPAS, Adult**
| 데이터셋 (건수) | 교차집단 | ① intersectional-DEO 전 → 후 | ② intersectional-DEO 전 → 후 |
|---|---|---|---|
| **COMPAS** (5,278) | 성별 × 인종 | 0.387 ± 0.105 → **0.129 ± 0.057** (5/5 개선) | 0.301 ± 0.067 → **0.108 ± 0.048** (5/5 개선) |
| **Adult** (48,842) | 성별 × 인종 | 0.503 ± 0.037 → **0.089 ± 0.054** (5/5 개선) | 0.470 ± 0.007 → **0.108 ± 0.035** (5/5 개선) |
| German Credit (1,000) | 성별 × 연령 | 0.464 → 0.430 (불안정) | 0.421 → 0.312 (시드 2개 악화) |

- 균형정확도 변화: ① COMPAS 0.661→0.642, Adult 0.745→0.710 / ② COMPAS 0.663→0.649, Adult 0.660→0.716(상승).
- **German Credit은 적용 대상에서 제외합니다.** 1,000건이라 교차셀당 양성 표본이 너무 적어, ①은 일부 시드에서 안전장치
  (셀 양성 20건 미만이면 전체 기준선 사용)로 보정이 일어나지 않고 ②는 시드마다 결과가 크게 흔들립니다.
- 즉 **보호속성·정답 라벨을 갖추고 교차집단별 양성 사례가 충분한(셀당 수십 건 이상) 정형 분류 데이터**에 적용 가능합니다.
- 상세: `results/multiseed_{itc,sbr}[_adult|_german].json` / `.log` (COMPAS는 접미사 없음).

### 이미지 데이터 — ① 이미지 연동 (SIIM-ACR 흉부 X선 12,047장, 교차집단 = 성별 × 65세 기준)
가이드 이미지 처리 흐름을 따릅니다: 데이터셋 클래스 → 특징 벡터를 DataFrame으로 → `aifData` → 분할 → LR baseline →
① 보정 → 결과를 `aifData`로 → `common_utils.compute_metrics` → `run()`이 `metrics_orig`, `metrics_transf` 반환.

| 특징 (`feature=`) | 분할 | 균형정확도 | intersectional-DEO 전 → 후 |
|---|---|---|---|
| `'pixel'` (64×64 픽셀 펼침, 가이드 예시 그대로) | 60/20/20 | 0.573 ± 0.013 | 0.132 ± 0.059 → 0.158 ± 0.041 |
| `'raddino'` (기본값, 고정된 RAD-DINO 임베딩) | 50/25/25 | 0.824 ± 0.006 | 0.082 ± 0.040 → 0.094 ± 0.039 |

**이미지에서는 공정성 개선이 재현되지 않았습니다.** 그대로 보고합니다.
- 픽셀 특징: baseline LR이 거의 판별을 못 해(균형정확도 0.57) 집단별 기준선을 다르게 잡을 근거가 없습니다.
- RAD-DINO 특징: baseline은 좋아졌지만(0.82) **보정 전부터 네 셀의 민감도가 0.85~0.89로 고릅니다.** 이 셀 크기(양성 52~319)에서는
  실제 격차가 0이어도 우연만으로 intersectional-DEO가 평균 0.073(95% 상한 0.136) 나오므로, 0.082는 우연 수준입니다.
  **SIIM에는 측정 가능한 교차집단 편향이 없어 보정이 줄일 대상이 없는 것**입니다.
- 이미지 모듈은 ①을 이미지 데이터 파이프라인에 연결하는 용도로 제공하며, 별도 알고리즘으로 세지 않습니다.
- 가이드 예시와의 차이: 기본 특징은 픽셀 대신 RAD-DINO 임베딩입니다(`feature='pixel'`로 가이드 예시와 동일하게 실행 가능).
- 상세: `results/multiseed_itc_image.*`(RAD-DINO), `results/multiseed_itc_image_pixel.*`(픽셀). 셀별 양성 수·TPR·임계값 포함.

재현: `python tools/multiseed.py --algo itc|sbr [--dataset adult|german]`, `--algo itc_image|itc_image_pixel`.

---

## 데이터
| 데이터 | 받는 곳 | 비고 |
|---|---|---|
| COMPAS | `https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv` | `download_data.py`가 AIF360 폴더로 저장 |
| German Credit | UCI `statlog/german/german.data`, `german.doc` | 〃 |
| Adult | UCI `adult/adult.data`, `adult.test`, `adult.names` | 〃 |
| SIIM-ACR 영상·전문가 마스크 (512×512 PNG) | Kaggle `iafoss/siimacr-pneumothorax-segmentation-data-512` | Kaggle 표기 라이선스: unknown — 재배포하지 않고 받는 방법만 안내 |
| SIIM-ACR 성별·나이 (DICOM 헤더 추출) | Kaggle `safavieh/siim-acr-pneumothorax-segmentation-metadata` | `train.csv`→`meta_train.csv`, `test.csv`→`meta_test.csv` |

SIIM 폴더 구조 (`./Sample/siim` 또는 `$SIIM_ROOT`):
```
iafoss512/train/<uid>.png    iafoss512/masks/<uid>.png    meta_train.csv    meta_test.csv
```
`download_data.py --siim`이 위 구조로 받습니다(Kaggle CLI 설치 및 `~/.kaggle` API 토큰 필요).
라벨·성별·나이 표(`siim_manifest.csv`)는 `to_dataset()`가 처음 실행 시 `tools/build_siim_manifest.py`로 만듭니다
(label = 전문가 마스크에 양성 픽셀이 있으면 1).

- **분할 단위**: 공개 SIIM 메타에는 별도 PatientID가 없고 `uid`(StudyInstanceUID, 스터디당 1장)가 가장 세밀한 단위라
  study 단위로 분할합니다. 코드에서 `uid` 중복을 제거하고 유일성을 확인합니다.
- **성별 포맷**: SIIM 메타의 `PatientSex`는 문자열 `'M'`/`'F'`입니다. 이미지 모듈은 그 외 값이 있으면 오류를 냅니다.

---

## 재현성
- 버전: `environment.txt` (torch 2.3.0+cu121, transformers 4.44.2, scikit-learn 1.7.2, numpy 2.2.6, aif360 0.6.1, Pillow 12.3.0),
  RAD-DINO revision `110cbc18d5133582e320b43d53bf5c44e410c936` (코드에 고정).
- 모든 알고리즘은 `seed` 인자로 분할을 결정하며 기본값은 1(가이드 규칙).

## MAF 통합 / 대상 버전
- 첨부받은 **MAF2024_guide** 양식 기준이며 `metrics/common_utils.py`를 자체 포함해 독립 실행됩니다.
- **MAF2025**가 지표를 `MAF.metric` 경로로 제공한다면 각 알고리즘의 `from metrics import common_utils`를 해당 경로로 바꾸고
  같은 `compute_metrics`에 연결하면 됩니다(AIF360 `ClassificationMetric` 기반으로 의미 동일). `intersectional_DEO`는 별도 계산합니다.
- 실제 연동 버전은 코난테크놀로지와 확인 후 확정합니다.

## experimental/ (미제출)
자기지도 백본(RAD-DINO) 임베딩 + 교차집단 보정(SIIM 5,000장, 성별×연령 6셀)을 시험했으나 5시드에서 개선이 재현되지 않아
(intersectional-DEO 0.264±0.094 → 0.284±0.091) 철회했습니다. 근거 로그와 분할 해시는 `experimental/repro/`에 있습니다.

## 라이선스
MIT (`LICENSE`). `metrics/common_utils.py`와 `aifData` 클래스는 MAF2024_guide 코드를 재사용했습니다.
