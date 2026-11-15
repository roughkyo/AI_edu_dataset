# 02. 학생 수학 성적 예측 데이터셋 (Student Performance Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.015–021

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 학생 수학 성적 예측 데이터셋 (Student Performance Dataset) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 회귀분석(Regression) / 지도학습 |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://archive.ics.uci.edu/dataset/320/student+performance |
| **원문 PDF 표기 링크** | https://archive.ics.uci.edu/ml/datasets/Student+Performance |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, UCI 신규 경로로 리다이렉트) |
| **교과서 연계** | 없음 |

학생들의 성별, 나이, 통학 시간, 공부 시간 등 8가지 속성과 수학 성적 간의 상관관계를 분석하는 회귀분석 데이터셋으로, 학생들의 실생활과 맞닿아 흥미도가 높습니다.

---

## 2. 데이터셋 특징

- 성별, 나이, 통학, 공부 시간, 이성 친구, 방과 후 자유시간, 건강, 결석, 수학 성적 등 **9개 속성**, **349개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: UCI 원본 데이터는 학교(`school`), 부모 교육수준, 1·2차 성적(`G1`, `G2`) 등을 포함해 총 33개 속성으로 이루어져 있으며, 포르투갈어 성적 데이터셋도 함께 제공됩니다. 본 자료집에서는 입문자 실습을 위해 9개 핵심 속성만 추출했습니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 성별 (`sex`) | 대상자의 성별 | 남자 `1`, 여자 `0` |
| ② | 나이 (`age`) | 대상자의 나이 | 최솟값 15, 최댓값 22 |
| ③ | 통학 (`traveltime`) | 집에서 학교까지 이동 시간 | `1`: 15분 미만, `2`: 15~30분, `3`: 30~60분, `4`: 60분 이상 |
| ④ | 공부 시간 (`studytime`) | 주당 공부 소요 시간 | `1`: 2시간 이하, `2`: 2~5시간, `3`: 5~10시간, `4`: 10시간 이상 |
| ⑤ | 이성 친구 (`romantic`) | 이성 친구 유무 | 있음 `1`, 없음 `0` |
| ⑥ | 방과 후 자유시간 (`freetime`) | 설문 기반 자유시간 수준 | `1`(매우 적음) ~ `5`(매우 많음) |
| ⑦ | 건강 (`health`) | 설문 기반 현재 건강 상태 | `1`(매우 나쁨) ~ `5`(매우 건강) |
| ⑧ | 결석 (`absences`) | 총 결석 일수 | 최솟값 0, 최댓값 93 |
| ⑨ | 수학 성적 (`G3`) | 3차 시험 기준 최종 수학 성적 | 최솟값 3, 최댓값 19 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 349행 × 9열의 가공 CSV 형태(성별, 나이, 통학시간, 공부시간, 이성친구, 방과후자유시간, 건강, 결석, 수학성적).
- **시각화 데이터**: 나이, 통학 시간, 공부 시간 등 각 열의 빈도 분포 막대그래프 제공.

---

## 5. 데이터셋 활용

1. **학습/테스트 데이터 분류**: 7:3 또는 8:2 비율 분할.
2. **각 속성 간 상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 공부 시간(입력) ↔ 수학 성적(출력) 상관관계 및 입출력 역전 비교
   - Q2: 통학 시간(입력) ↔ 수학 성적(출력) 상관관계 및 입출력 역전 비교
   - Q3: 이성 친구 유무·건강 상태 집단별 수학 성적 분포 차이 분석

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.019)에 수록된 개인 Google Drive 다운로드 링크(`1ROipU8RCoiw9gXl-78jOTT4BfjGyhg9Y`)는 현재 만료(HTTP 404)되었습니다. UCI 공식 페이지에서 `student.zip`을 내려받아 `student-mat.csv`(세미콜론 구분자 `;`)를 사용하시기 바랍니다.

```python
# [T02] 학생 수학 성적 예측 데이터셋 (Student Performance) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.018-021
# 주의: 원문 코드 내 Google Drive 링크는 만료(404)되었으므로 UCI 공식 링크에서 student-mat.csv를 내려받아 사용하세요.

import csv
import urllib.request
import pandas as pd


def main():
    # 1. 데이터 다운로드 (원문 Google Drive 링크는 만료되어 주석 처리)
    # url = "https://drive.google.com/u/3/uc?id=1ROipU8RCoiw9gXl-78jOTT4BfjGyhg9Y&export=download"
    # urllib.request.urlretrieve(url, "./math.csv")
    # 대체 공식 링크: https://archive.ics.uci.edu/dataset/320/student+performance (student-mat.csv)

    # 2. 내려받은 CSV 읽어오기 (UCI 원본은 sep=';' 사용)
    cols = [
        "school", "sex", "age", "address", "famsize", "Pstatus",
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian",
        "traveltime", "studytime", "failures", "schoolsup", "famsup",
        "paid", "activities", "nursery", "higher", "internet", "romantic",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences", "G1", "G2", "G3",
    ]
    df = pd.read_csv("./math.csv", names=cols)

    # 3. 활용하지 않는 속성 제거 (33개 -> 9개 속성 유지)
    drop_cols = [
        "school", "address", "famsize", "Pstatus", "Medu", "Fedu",
        "Mjob", "Fjob", "reason", "guardian", "failures", "schoolsup",
        "famsup", "paid", "activities", "nursery", "higher", "internet",
        "famrel", "goout", "Dalc", "Walc", "G1", "G2",
    ]
    df = df.drop(columns=drop_cols)

    # 4. 범주형 데이터 수치 변환 (M/F -> 1/0, yes/no -> 1/0)
    df.loc[df["sex"] == "M", "sex"] = 1
    df.loc[df["sex"] == "F", "sex"] = 0
    df.loc[df["romantic"] == "yes", "romantic"] = 1
    df.loc[df["romantic"] == "no", "romantic"] = 0

    # 5. 속성 명 한글 변환
    df.loc[0, "sex"] = "성별"
    df.loc[0, "age"] = "나이"
    df.loc[0, "traveltime"] = "통학 시간"
    df.loc[0, "studytime"] = "공부 시간"
    df.loc[0, "romantic"] = "이성 친구"
    df.loc[0, "freetime"] = "방과 후 자유시간"
    df.loc[0, "health"] = "건강"
    df.loc[0, "absences"] = "결석"
    df.loc[0, "G3"] = "수학 성적"

    # 6. CP949 저장 및 KNIME 호환 재저장
    df.to_csv("filename.csv", mode="w", index=False, header=False, encoding="CP949")

    with open("filename.csv", "r", encoding="CP949") as f:
        rdr = csv.reader(f)
        with open("math.csv", "w", newline="", encoding="CP949") as f2:
            wr = csv.writer(f2)
            for line in rdr:
                wr.writerow(line)


if __name__ == "__main__":
    main()
```
