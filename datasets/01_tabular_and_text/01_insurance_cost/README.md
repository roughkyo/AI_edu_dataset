# 01. 보험료 예측 데이터셋 (Medical Cost Personal Datasets)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.008–014

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 보험료 예측 데이터셋 (Medical Cost Personal Datasets) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 회귀분석(Regression) / 지도학습 |
| **난이도** | ★☆☆☆☆ (1/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://www.kaggle.com/datasets/mirichoi0218/insurance |
| **원문 PDF 표기 링크** | https://www.kaggle.com/mirichoi0218/insurance |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.70 (정규화), p.90 (학습데이터 시각화) |

대상자의 보험료 예측을 위한 데이터셋으로, 5가지 입력 요소(나이, 성별, BMI, 자녀 수, 흡연)와 보험료 간의 상관관계를 분석하는 회귀분석 입문용 데이터셋입니다.

---

## 2. 데이터셋 특징

- 정수로 나타내는 **나이(Age)**, 이분적 속성 **성별(Sex)**·**흡연(Smoker)**, 정량적 속성 **체질량지수(BMI)**·**자녀 수(Children)**·**보험료(Charges)** 등 총 **6개 속성**, **1,338개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 접근성 향상을 위해 원본의 `region` 속성을 제거하고, 달러(`$`) 단위 보험료를 원화(`₩`) 단위로 변환(100원 단위 버림)하는 가공이 포함되어 있습니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 나이 (`Age`) | 대상자의 나이 | 최솟값 18, 최댓값 64 |
| ② | 성별 (`Sex`) | 대상자의 성별 | `male` → `1`, `female` → `0` |
| ③ | 체질량지수 (`BMI`) | 키와 몸무게 기반 비만도 지수 | 최솟값 16, 최댓값 53.1 |
| ④ | 자녀 수 (`Children`) | 대상자의 자녀 수 | 최솟값 0, 최댓값 5 |
| ⑤ | 흡연 (`Smoker`) | 대상자의 흡연 여부 | `no` → `0`, `yes` → `1` |
| ⑥ | 보험료 (`Charges`) | 측정된 의료 보험료(원화 변환) | 최솟값 134,000원, 최댓값 7,652,000원 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 1,338행 × 6열(원본은 `region` 포함 7열)의 CSV 표 형태이며, 각 행이 한 대상자를 나타냅니다.
- **시각화 데이터**: 각 열별 막대/분포 히스토그램을 통해 나이(18~64), 성별 비율(male 51%, female 49%), BMI, 자녀 수(0~5) 분포를 직관적으로 확인할 수 있습니다.

---

## 5. 데이터셋 활용

1. **데이터셋 정규화(Normalization)**: 단위와 값의 범위가 다른 속성들에서 평균을 빼고 표준편차로 나누어 평균 0, 표준편차 1 근처로 변환 (`인공지능과 미래사회 p.70`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 비율로 분할해 학습에 쓰지 않은 테스트 데이터로 성능을 평가.
3. **학습데이터 시각화**: 막대그래프로 속성별 분포 확인 (`인공지능과 미래사회 p.90`).
4. **각 요소 간 상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 나이 ↔ BMI 상관관계 및 입출력 역전 비교
   - Q2: 자녀 수 ↔ 보험료 상관관계 및 입출력 역전 비교
   - Q3: 성별·흡연 여부 집단 간 보험료 분포 차이 분석

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.012~014)에서 `insurance['sex'][df['sex'] == ...]`로 미정의 변수 `df`를 참조한 오타와 MATLAB 스타일 `%%` 주석을 실행 가능한 Python 문법(`insurance` 변수 통일, `#` 주석)으로 정리했습니다. 코드 내 GitHub raw CSV URL은 현재도 정상 작동(200 OK)합니다.

```python
# [T01] 보험료 예측 데이터셋 (Medical Cost Personal Datasets) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.012-014
# 참고: 원문 코드의 미정의 변수(df -> insurance) 오타를 수정함

import csv
import urllib.request
import numpy as np
import pandas as pd


def main():
    # 1. 데이터 다운로드 (GitHub 공개 미러 - 200 OK 확인됨)
    print("Beginning file download with urllib2...")
    url = "https://raw.githubusercontent.com/krish1407/Medical-Cost-Personal-Datasets/master/insurance.csv"
    urllib.request.urlretrieve(url, "./insurance.csv")

    # 2. 내려받은 CSV 읽어오기 (첫 행이 헤더이므로 skiprows=1 또는 header=0 활용)
    insurance = pd.read_csv(
        "./insurance.csv",
        names=["age", "sex", "bmi", "children", "smokes", "region", "charges"],
        encoding="UTF-8",
    )

    # 3. 성별 및 흡연 여부 수치 라벨 변환 (male:1, female:0 / yes:1, no:0)
    # 원문 코드에서는 df['sex']로 오기되었으나 insurance['sex']가 올바름
    insurance.loc[insurance["sex"] == "male", "sex"] = 1
    insurance.loc[insurance["sex"] == "female", "sex"] = 0
    insurance.loc[insurance["smokes"] == "yes", "smokes"] = 1
    insurance.loc[insurance["smokes"] == "no", "smokes"] = 0

    # 4. $(달러) 단위를 ₩(원) 단위로 변환 (120배 적용 후 천원 미만 버림)
    for i in range(insurance.shape[0] - 1):
        dollar = float(insurance["charges"].iloc[i + 1])
        won = str(int(np.round(dollar * 120) - np.round(dollar * 120) % 1000))
        insurance.loc[insurance.index[i + 1], "charges"] = won

    # 5. 지역 속성 제거 및 한글 속성명 변환
    insurance = insurance.drop("region", axis=1)
    insurance.loc[0, "age"] = "나이"
    insurance.loc[0, "sex"] = "성별"
    insurance.loc[0, "bmi"] = "체질량지수"
    insurance.loc[0, "children"] = "자녀 수"
    insurance.loc[0, "smokes"] = "흡연"
    insurance.loc[0, "charges"] = "보험료"

    # 6. CP949 저장 및 KNIME 등 노코딩 플랫폼 호환 재저장
    insurance.to_csv("temp.csv", mode="w", index=False, header=False, encoding="CP949")

    with open("temp.csv", "r", encoding="CP949") as f1:
        r_temp = csv.reader(f1)
        with open("보험료.csv", "w", newline="", encoding="CP949") as f2:
            wr = csv.writer(f2)
            for line in r_temp:
                wr.writerow(line)


if __name__ == "__main__":
    main()
```
