# MAF 추가 알고리즘 — KAIST 유창동 연구실 (AI-DEP 의료3 / ALINPneumothorax)

MSIT AI FAIR(MAF) 프레임워크에 추가할 **공정성(Fairness) 알고리즘**.
AIF360 기반 MAF2024_guide 양식(`aifData` + Logistic Regression baseline +
`metrics.common_utils`)에 맞춰 작성, 정형 벤치마크(COMPAS/German/Adult)에서 실행·검증.
두 알고리즘 모두 **교차집단(intersectional, 예: 성별×인종)** 공정성을 다루는 것이 특징입니다.

> **최종 제출 알고리즘 수: 2종**(아래 ①②). ③은 실험적(미제출) — `experimental/` 참조.

---

## 쉬운 설명 (처음 보는 분을 위해)

의료·의사결정 AI가 "양성/음성"을 잘 맞혀도, **특정 집단(예: 고령 남성, 특정 성별×인종)에게만
자주 틀리면** 공정하지 않습니다. 아래 알고리즘은 그 집단 간 격차를 줄입니다. 특히 "성별만",
"나이만" 보면 괜찮아 보여도 **둘을 겹쳐 보면 드러나는 사각지대(교차집단)** 를 다룹니다.

- **① 교차집단 사후 임계값 보정** — AI는 보통 "점수 0.5 이상이면 양성"이라는 **하나의 기준선을
  모두에게 똑같이** 적용합니다. 그러면 어떤 집단은 유독 많이 놓칩니다. 이 방법은 **집단마다
  기준선을 조금씩 다르게** 맞춰 모든 집단이 비슷한 비율로 양성을 잡게 합니다. **모델 재학습
  없이** 판정 기준만 바꿉니다. (예측 '이후' 단계)
- **② 교차집단 균형 리샘플링** — AI는 학습 때 **많이 본 집단은 잘, 적게 본 집단은 못** 맞힙니다.
  데이터에 적은 집단의 사례를 **복제해 더 자주 보여주어** 골고루 학습시킵니다. (학습 '이전' 단계)

---

## ① ② 제출 알고리즘

| # | 범주 | 파일 |
|---|------|------|
| ① | post-processing | `algorithms/postprocessing/IntersectionalThresholdCalibration.py` |
| ② | pre-processing | `algorithms/preprocessing/SubgroupBalancedResampling.py` |

**①** 학습된 분류기의 점수에 대해 성별×인종(또는 연령×성별) 등 **교차집단별 등가기회
(equal-opportunity) 임계값**을 조정, 재학습 없이 집단 간 TPR 격차를 보정. 희소 셀은 전역 임계값
폴백(MIN_POS). **②** (라벨×교차집단) 그룹을 빈도 역가중 **오버샘플링**하여 학습데이터 집단
불균형 편향을 완화.

새 지표: **intersectional-DEO** = (교차집단 TPR의 최대−최소), 낮을수록 공정. AIF360 표준 지표
(`equal_opportunity_difference`, `average_abs_odds_difference` 등)도 주 보호속성 기준으로 보고.

### 검증 (COMPAS, 성별×인종 4개 교차셀, 5개 split seed)
| 알고리즘 | intersectional-DEO (보정 전 → 후) |
|---|---|
| ① IntersectionalThresholdCalibration | **0.387 ± 0.105 → 0.129 ± 0.057** (5시드 전부 개선) |
| ② SubgroupBalancedResampling | **0.301 ± 0.067 → 0.108 ± 0.048** (5시드 전부 개선) |

(정확도는 소폭 하락 — 공정성 개선의 통상적 트레이드오프.) 단일 실행 재현:
```bash
python algorithms/postprocessing/IntersectionalThresholdCalibration.py   # ①
python algorithms/preprocessing/SubgroupBalancedResampling.py            # ②
```

---

## ③ 실험적 (미제출) — `experimental/`

**자기지도 백본 + 교차집단 보정** (흉부 X선). 자기지도(RAD-DINO/DINOv2) 임베딩 위에 분류기를
학습하고 ①의 교차집단 보정을 적용하는 이미지 파이프라인.

**정직한 결과 — 공정성 알고리즘으로 제출하지 않음:**
- 자기지도 백본은 **성능**을 크게 높입니다(표현학습). 그러나 **5개 시드로 검증한 결과, 이미지
  규모(SIIM 5,000장)에서 교차집단 보정은 공정성을 안정적으로 개선하지 못했습니다**:
  intersectional-DEO **0.264±0.094 → 0.284±0.091** (평균적으로 개선 없음/소폭 악화), gender-DEO
  0.042±0.040 → 0.048±0.048. 원인: test 교차셀 양성수가 13~67개로 작아 검증셋에서 적합한
  임계값이 test에 일반화되지 않음(소표본 과적합).
- 즉 **자기지도 학습 자체는 공정성 기법이 아니며**(성능 요소), 교차집단 보정은 ①②처럼 **셀이
  충분한 정형 데이터에서만 안정적으로 작동**합니다. 근거 로그: `experimental/repro/`.

---

## 실행 환경 & 데이터 준비
- Python 3.10, Linux, (③은 CUDA). 설치: `pip install -r requirements.txt`. 정확한 버전: `environment.txt`.
- **AIF360 벤치마크**(COMPAS/German)는 라이선스상 원본 미동봉 — 아래를 받아 지정 폴더에:
  (`<AIF360>` = `python -c "import aif360,os;print(os.path.dirname(aif360.__file__))"`)
  - COMPAS: `https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv` → `<AIF360>/data/raw/compas/`
  - German: `https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data` (+ `.doc`) → `<AIF360>/data/raw/german/`

---

## 재현성 (reproducibility)
- **패키지 버전 / 모델 revision**: `environment.txt` — torch 2.3.0+cu121, transformers 4.44.2,
  scikit-learn 1.7.2, numpy 2.2.6, aif360 0.6.1, Pillow 12.3.0 / **microsoft/rad-dino @ `110cbc18d5133582e320b43d53bf5c44e410c936`**.
- **SIIM manifest 생성 코드**: `tools/build_siim_manifest.py` — images/masks/meta로부터 `uid,label,sex,age,view`
  생성. label=전문가 마스크에 양성 픽셀 존재 여부. (기존 manifest와 label 600/600·sex·age 100% 일치 확인.)
- **분할(split)**: ③의 `--emit_split`이 train/valid/test의 **uid 목록 + 각 split의 SHA-256**을 기록
  (`experimental/repro/siim_split_seed1.json`). ①②는 AIF360 `split(..., seed=self.seed)`로 결정적.
- **환자 중복 차단**: 공개 SIIM-ACR 메타에는 **별도 PatientID가 없고**, 행 식별자 `uid =
  StudyInstanceUID`(스터디당 1장)가 가장 세밀한 단위입니다. 따라서 분할은 **study(uid) 단위**이며,
  코드에서 `uid` 중복을 제거(`drop_duplicates`)하고 유일성을 assert하여 같은 스터디가 두 split에
  들어가지 않게 합니다. 공개데이터 한계상 그보다 세밀한 환자 단위 분할은 불가합니다.
- **성별 포맷**: manifest의 `sex`는 **문자열 'M'/'F'** 를 기대합니다(SIIM 원 포맷). 숫자 0/1(AIF360
  관례: 1=남성)도 허용하도록 파서를 보강했고, 단일 값으로 붕괴하면 에러를 발생시킵니다
  (과거 "숫자 0/1이 모두 0으로 변환"되던 버그 수정).

---

## MAF 통합 / 대상 버전
- 본 패키지는 **첨부받은 MAF2024_guide**(aifData + LR baseline + `metrics.common_utils`) 양식에
  맞춰 작성했고, `metrics/common_utils.py`를 **자체 포함**하므로 독립 실행됩니다.
- **MAF2025**가 지표를 `MAF.metric` 경로로 제공한다면, 각 알고리즘의 `from metrics import
  common_utils`를 해당 경로로 교체하고 동일 함수(`compute_metrics`)에 매핑하면 됩니다. 지표 의미는
  동일(AIF360 `ClassificationMetric` 기반)이며, 추가한 `intersectional_DEO`는 다중 교차셀의 max−min TPR로
  별도 계산됩니다. 실제 대상 버전/연동은 코난테크놀로지와 확인 후 확정 예정.

## 라이선스
MIT (`LICENSE`). 과제/연구실 정책에 따라 변경 가능.

## 출처
KAIST 유창동 연구실 · AI-DEP 의료3(ALINPneumothorax). `metrics/common_utils.py`는 MAF2024_guide 재사용.
