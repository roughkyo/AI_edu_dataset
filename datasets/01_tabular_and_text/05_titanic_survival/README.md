# 05. 타이타닉 생존자 예측 데이터셋 (Titanic - Machine Learning from Disaster)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.036–042

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 타이타닉 생존자 예측 데이터셋 (Titanic - Machine Learning from Disaster) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 분류(Classification - 이진 분류) / 지도학습 |
| **난이도** | ★☆☆☆☆ (1/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://www.kaggle.com/c/titanic |
| **원문 PDF 표기 링크** | https://www.kaggle.com/c/titanic |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle Competition 공식 페이지) |
| **교과서 연계** | 없음 |

타이타닉 호 탑승자의 입장권 등급, 성별, 나이, 가족 수, 요금, 승선 항구 정보를 바탕으로 생존 여부를 예측하는 대표적 이진 분류 입문 데이터셋입니다.

---

## 2. 데이터셋 특징

- 총 **8개 속성**으로 구성되며, 나이 결측치 제거 등의 정제 과정을 거쳐 자료집 기준 **434개**(예시 표 설명에는 714행 언급)의 데이터 포인트를 활용합니다.
- **원본 데이터와의 차이**: 원본 10개(PassengerId, Name, Ticket, Cabin 포함)에서 식별자·문자열 4개 속성을 제거하고, 운임(`Fare`)을 원화 단위로 변환, 나이(`Age`)를 반올림 정수화하며 생존 여부(`Survived`) 열을 맨 뒤로 재배치했습니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 입장권 등급 (`Pclass`) | 승차권 등급 | `1`: 1등급, `2`: 2등급, `3`: 3등급 |
| ② | 성별 (`Sex`) | 대상자 성별 | `1`: 남자, `0`: 여자 |
| ③ | 나이 (`Age`) | 대상자 나이(반올림 정수) | 최솟값 0, 최댓값 81 |
| ④ | 탑승 가족(형제) (`SibSp`) | 함께 탄 형제/배우자 수 | 최솟값 0, 최댓값 5 |
| ⑤ | 탑승 가족(부모, 자식) (`Parch`) | 함께 탄 부모/자식 수 | 최솟값 0, 최댓값 6 |
| ⑥ | 입장권 요금 (`Fare`) | 원화 변환 탑승권 요금 | 최솟값 0, 최댓값 614,000 |
| ⑦ | 승선 항구 (`Embarked`) | 탑승 항구 | `0`: S(사우스햄튼), `1`: C(셰르부르), `2`: Q(퀸즈타운) |
| ⑧ | 생존 여부 (`Survived`) | 생존 여부(목표 변수) | `1`: 생존, `0`: 사망 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 결측치(`Age`) 제거 후 구성된 CSV 표.
- **시각화 데이터**: Pclass, Sex, Age, SibSp 분포 시각화 및 등급별 생존자/사망자 막대그래프 제공.

---

## 5. 데이터셋 활용

1. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
2. **생존 여부 분류 탐구 (Q1~Q3)**:
   - Q1: 나이 입력 vs 다중 속성 입력 성능 비교
   - Q2: 입장권 등급 입력 vs 다중 속성 조합 성능 비교
   - Q3: 입장권 등급, 나이 등 집단별 생존율 분포 차이 분석

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.039)의 `blog.kakaocdn.net` 링크는 현재 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.csv`를 내려받아 사용하세요.

```python
# [T05] 타이타닉 생존자 예측 데이터셋 (Titanic) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.039-042
# 주의: 원문 내 kakaocdn 링크는 만료(404)되었으므로 Kaggle에서 train.csv를 내려받아 사용하세요.

import csv
import urllib.request
import numpy as np
import pandas as pd


def main():
    # 1. 데이터 다운로드 (원문 kakaocdn 링크 만료로 주석 처리)
    # url = "https://blog.kakaocdn.net/dn/3rvGg/btqB3XNjWym/F0nUXZeecI8eoZX-L2MMWn0/train.csv?attach=1&knm=tfile.csv"
    # urllib.request.urlretrieve(url, "./titanic.csv")
    # 대체 공식 링크: https://www.kaggle.com/c/titanic

    # 2. 내려받은 CSV 읽어오기
    cols = [
        "PassengerId", "Survived", "Pclass", "Name", "Sex",
        "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
    ]
    df = pd.read_csv("./titanic.csv", names=cols, encoding="UTF-8")

    # 3. 성별 및 승선 항구 수치 라벨 변환
    df.loc[df["Sex"] == "male", "Sex"] = 1
    df.loc[df["Sex"] == "female", "Sex"] = 0
    df.loc[df["Embarked"] == "S", "Embarked"] = 0
    df.loc[df["Embarked"] == "C", "Embarked"] = 1
    df.loc[df["Embarked"] == "Q", "Embarked"] = 2

    # 4. 불필요한 속성 제거
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # 5. 나이 반올림 및 운임($ -> ₩) 변환
    for i in range(df.shape[0] - 1):
        idx = df.index[i + 1]
        if pd.notna(df.loc[idx, "Age"]):
            df.loc[idx, "Age"] = int(np.round(float(df.loc[idx, "Age"])))
        dollar = float(df.loc[idx, "Fare"])
        won = int(np.round(dollar * 1200) - np.round(dollar * 1200) % 1000)
        df.loc[idx, "Fare"] = won

    # 6. 결측치 제거 및 생존 여부(Survived) 열 맨 뒤로 재배치
    df = df.dropna(subset=["Age"])
    df = df.reindex(columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"])

    # 7. 속성 명 한글 변환
    kr_names = {
        "Survived": "생존 여부", "Pclass": "입장권 등급", "Sex": "성별",
        "Age": "나이", "SibSp": "탑승 가족(형제)", "Parch": "탑승 가족(부모, 자식)",
        "Fare": "입장권 요금", "Embarked": "승선 항구",
    }
    for k, v in kr_names.items():
        df.loc[0, k] = v

    # 8. CP949 저장 및 KNIME 호환 재저장
    df.to_csv("filename.csv", mode="w", index=False, header=False, encoding="CP949")

    with open("filename.csv", "r", encoding="CP949") as f:
        rdr = csv.reader(f)
        with open("titanic.csv", "w", newline="", encoding="CP949") as f2:
            wr = csv.writer(f2)
            for line in rdr:
                wr.writerow(line)


if __name__ == "__main__":
    main()
```
