# 03. 포도주 품질 예측 데이터셋 (Red Wine Quality)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.022–028

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 포도주 품질 예측 데이터셋 (Red Wine Quality) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 회귀분석(Regression) / 지도학습 |
| **난이도** | ★☆☆☆☆ (1/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009 |
| **원문 PDF 표기 링크** | https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.70 (정규화) |

적포도주의 화학적 성분(구연산, 잔류 당분, 염화나트륨, 밀도, pH, 황산칼륨, 도수)과 포도주 품질 점수 간의 상관관계를 다루는 회귀분석(및 분류 변환 가능) 데이터셋입니다.

---

## 2. 데이터셋 특징

- 7가지 이화학적 속성 + 품질(`quality`) 등 총 **8개 속성**, **1,599개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 원본 데이터는 고정산도(`fixed acidity`), 휘발성 산도(`volatile acidity`) 등을 포함한 총 12개 속성이며, 실습 편의를 위해 4개 속성을 제외하고 8개 속성만 사용합니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 |
|---|---|---|---|
| ① | 구연산 농도 (`citric acid`) | 포도주의 구연산 농도 | 최솟값 0, 최댓값 1 |
| ② | 잔류 당분 농도 (`residual sugar`) | 발효 후 남은 당분 농도 | 최솟값 0.9, 최댓값 15.5 |
| ③ | 염화나트륨 농도 (`chlorides`) | 소금 성분 농도 | 최솟값 0.012, 최댓값 0.611 |
| ④ | 밀도 (`density`) | 알코올 도수와 당분 함량의 영향을 받는 밀도 | 최솟값 0.990, 최댓값 1.003 |
| ⑤ | pH | 수소 이온 농도 지수(산성/알칼리성) | 최솟값 2.74, 최댓값 4.01 |
| ⑥ | 황산칼륨 농도 (`sulphates`) | 황산칼륨(방부/산화방지) 농도 | 최솟값 0.33, 최댓값 2 |
| ⑦ | 도수 (`alcohol`) | 알코올 도수 | 최솟값 8.4, 최댓값 14.9 |
| ⑧ | 품질 (`quality`) | 포도주 품질 점수 | 0(나쁨) ~ 10(좋음) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 1,599행 × 8열(구연산, 잔류 당분, 염화나트륨, 밀도, pH, 황산칼륨, 도수, 품질)의 CSV 표.
- **시각화 데이터**: Kaggle 시각화 기준 각 성분별 히스토그램 제공.

---

## 5. 데이터셋 활용

1. **데이터셋 정규화(Normalization)** (`인공지능과 미래사회 p.70`)
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 도수(입력) ↔ 밀도(출력) 상관관계 및 역전 비교
   - Q2: 잔류 당분(입력) ↔ 품질(출력) 상관관계 및 역전 비교
   - Q3: 계절·평일 여부 등에 따른 비교(원문 Q3 문구 일부는 T04와 혼재)
4. **회귀분석 문제를 분류 문제로 변환해보기**: 품질 0~5점을 `0`(나쁨), 6점 이상을 `1`(좋음)으로 이진화하거나 3단계로 범주화해 분류 모델 실험.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.026)의 Google Drive 링크(`18w9sj9bwY7nU1z51nqCg3bfAcAU7C-qXA`)는 현재 만료(404)되었으며, Kaggle 공식 링크에서 `winequality-red.csv`를 내려받아 사용할 수 있습니다.

```python
# [T03] 포도주 품질 예측 데이터셋 (Red Wine Quality) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.026-028
# 주의: 원문 코드 내 Google Drive 링크는 만료(404)되었으므로 Kaggle 공식 링크에서 winequality-red.csv를 내려받아 사용하세요.

import csv
import urllib.request
import pandas as pd


def main():
    # 1. 데이터 다운로드 (원문 Google Drive 링크 만료로 주석 처리)
    # url = "https://drive.google.com/u/3/uc?id=18w9sj9bwY7nU1z51nqCg3bfAcAU7C-qXA&export=download"
    # urllib.request.urlretrieve(url, "./wine.csv")
    # 대체 공식 링크: https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

    # 2. 내려받은 CSV 읽어오기
    cols = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
        "pH", "sulphates", "alcohol", "quality",
    ]
    df = pd.read_csv("./wine.csv", names=cols)

    # 3. 사용하지 않는 4개 속성 제거 (12개 -> 8개)
    df = df.drop(["fixed acidity", "volatile acidity", "free sulfur dioxide", "total sulfur dioxide"], axis=1)

    # 4. 속성 명 한글 변환
    df.loc[0, "citric acid"] = "구연산 농도"
    df.loc[0, "residual sugar"] = "잔류 당분 농도"
    df.loc[0, "chlorides"] = "염화나트륨 농도"
    df.loc[0, "density"] = "밀도"
    df.loc[0, "pH"] = "pH"
    df.loc[0, "sulphates"] = "황산칼륨 농도"
    df.loc[0, "alcohol"] = "도수"
    df.loc[0, "quality"] = "품질"

    # 5. CP949 저장 및 KNIME 호환 재저장
    df.to_csv("filename.csv", mode="w", index=False, header=False, encoding="CP949")

    with open("filename.csv", "r", encoding="CP949") as f:
        rdr = csv.reader(f)
        with open("wine.csv", "w", newline="", encoding="CP949") as f2:
            wr = csv.writer(f2)
            for line in rdr:
                wr.writerow(line)


if __name__ == "__main__":
    main()
```
