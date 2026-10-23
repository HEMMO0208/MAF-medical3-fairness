# Review Response — MAF 의료3 제출 검증 대응

AI-DEP 10/23 의료3 제출(KAIST 유창동 연구실) 검증 지적사항에 대한 보완 내역입니다.
기준 커밋: 검토본 `3a028bf` → **보완본 `477d76e` 이후**.

## 요약 (headline)
- 리뷰어 요청대로 **5개 시드 멀티시드 검증**을 수행했습니다.
- 그 결과 **③(자기지도 백본 + 교차집단 보정)은 이미지 규모에서 공정성 개선이 재현되지 않아**,
  공정성 알고리즘에서 **철회**하고 `experimental/`에 **네거티브 결과**로 공개했습니다.
- **최종 제출 알고리즘 = 2종(① ②)**, 둘 다 5시드에서 견고하게 교차집단 격차를 줄입니다.

---

## 1. SIIM-ACR 재현성 (데이터/manifest/분할/로그/버전/revision)
- **데이터 접근·전처리**: 공개 SIIM-ACR Pneumothorax. images=`iafoss512/train/<uid>.png`,
  masks=`iafoss512/masks/<uid>.png`, demographics=`meta_{train,test}.csv`(DICOM 유래). 자세한 절차는
  `README.md > 재현성`, 생성 코드는 `tools/build_siim_manifest.py`.
- **manifest 생성 코드**: `tools/build_siim_manifest.py` — `uid,label,sex,age,view` 생성.
  `label` = 전문가 마스크에 양성 픽셀 존재 여부. **기존 manifest와 대조 검증: label 600/600 일치,
  sex·age 100% 일치(meta 매칭 10,675건).**
- **train/valid/test 분할 목록·해시**: `experimental/repro/siim_split_seed1.json` — 각 split의 **uid 전체
  목록 + SHA-256**. (예: test SHA-256 = `195b1813dfac7f6b...`)
- **원 실행 로그**: `experimental/repro/image_experiment_run.log` (5시드 전체 출력).
- **패키지 버전**: `environment.txt` — torch 2.3.0+cu121, transformers 4.44.2, scikit-learn 1.7.2,
  numpy 2.2.6, pandas, aif360 0.6.1, Pillow 12.3.0.
- **RAD-DINO revision**: `microsoft/rad-dino @ 110cbc18d5133582e320b43d53bf5c44e410c936` (코드에 고정).

## 2. 환자 중복 차단 (무작위 이미지 분할 관련)
- 공개 SIIM-ACR 메타에는 **별도 PatientID가 없습니다.** 가장 세밀한 식별자는 `uid =
  StudyInstanceUID`(스터디당 1장)입니다. 따라서 분할은 **study(uid) 단위**로 수행합니다.
- 코드는 `uid` **중복을 제거**(`drop_duplicates(subset='uid')`)하고 **유일성을 assert**하여, 같은
  스터디가 두 split에 동시에 들어가지 않도록 보장합니다.
- 공개 데이터 한계상 study보다 세밀한 **환자 단위** 분할은 불가능하며, 이 점을 README에 명시합니다.

## 3. 성별/교차 결과 동시 보고 + 셀별 양성수·TPR + 멀티시드
- **③ 5시드 결과(SIIM 5,000장, 성별×연령 6셀)**:
  - intersectional-DEO: **0.264 ± 0.094 → 0.284 ± 0.091** (평균 개선 없음/소폭 악화)
  - gender-DEO: 0.042 ± 0.040 → 0.048 ± 0.048
  - 셀별 양성수(예시 seed=1): Fx<50=50, Fx50-64=28, Fx65+=22, Mx<50=67, Mx50-64=44, Mx65+=13
    (셀별 TPR before/after는 `experimental/repro/image_experiment_run.log`에 수록)
  - → 소표본(셀당 13~67 양성) 과적합으로 **일반화 실패** → ③ 철회.
- **제출본 ①② 5시드(COMPAS)**: 모두 전 시드에서 개선.
  - ① intersectional-DEO **0.387 ± 0.105 → 0.129 ± 0.057**
  - ② intersectional-DEO **0.301 ± 0.067 → 0.108 ± 0.048**

## 4. 대상 MAF 버전 / 이미지 모듈 지표 매핑
- 본 패키지는 **첨부받은 MAF2024_guide** 양식(`aifData` + LR baseline + `metrics.common_utils`)에
  맞춰 작성, `metrics/common_utils.py`를 **자체 포함**하여 독립 실행됩니다.
- **MAF2025가 `MAF.metric` 경로**를 쓸 경우: 각 알고리즘의 `from metrics import common_utils`를 해당
  경로로 교체하고 동일 함수(`compute_metrics`)에 매핑하면 됩니다. 지표 의미는 동일(AIF360
  `ClassificationMetric` 기반). 추가 지표 `intersectional_DEO`는 다중 교차셀의 max−min TPR로 별도 계산.
- 실제 대상 버전/연동은 코난테크놀로지와 확인 후 확정 예정입니다.

## 5. LICENSE / 제출 알고리즘 수 / 성별 포맷
- **LICENSE**: MIT(`LICENSE`) 추가. 과제/연구실 정책에 따라 변경 가능.
- **최종 제출 알고리즘 수 = 2종**(① post-processing, ② pre-processing). 기존 README의 "2종" 문단과
  파일 3종 불일치는, ③을 `experimental/`(미제출)로 분리하여 해소했습니다.
- **성별 포맷**: manifest `sex`는 **문자열 'M'/'F'** 를 기대합니다(SIIM 원 포맷). 숫자 0/1(AIF360
  관례상 1=남성)도 처리하도록 파서를 보강했고, 단일 값으로 붕괴 시 오류를 발생시킵니다.
  (지적하신 "숫자 0/1 입력 시 전부 0으로 변환"되던 버그를 수정.)

---
문의: KAIST 유창동 연구실 · AI-DEP 의료3(ALINPneumothorax).
