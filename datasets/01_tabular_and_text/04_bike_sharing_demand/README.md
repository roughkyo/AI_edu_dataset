# 04. 자전거 대여량 예측 데이터셋 (Bike Sharing Demand)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.029–035

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 자전거 대여량 예측 데이터셋 (Bike Sharing Demand) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 회귀분석(Regression) / 지도학습 |
| **난이도** | ★☆☆☆☆ (1/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://www.kaggle.com/c/bike-sharing-demand |
| **원문 PDF 표기 링크** | https://www.kaggle.com/c/bike-sharing-demand |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle Competition 공식 페이지) |
| **교과서 연계** | 없음 |

날씨, 시간, 휴일 등 12가지 속성과 자전거 대여량(비회원/회원/총 대여) 간의 상관관계를 분석하는 회귀분석 데이터셋입니다.

---

## 2. 데이터셋 특징

- 시간, 계절, 휴일, 평일, 날씨, 기온, 체감온도, 습도, 풍속, 비회원 대여, 회원 대여, 총 대여 등 **12개 속성**, **10,886개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 날짜+시간 문자열(`datetime`)에서 날짜를 제거하고 시간(`0~23`)만 추출했으며, 기온·체감온도·풍속을 반올림해 정수로 변환했습니다.
- **편집 참고**: p.029 도입문 첫 문장에 "총 9가지의 속성"이라는 오기(실제는 12개 속성)가 섞여 있습니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 시간 (`datetime`) | 당일 시간(원본 날짜 제거) | `0` (자정) ~ `23` (오후 11시) |
| ② | 계절 (`season`) | 계절 구분 | `1`: 봄, `2`: 여름, `3`: 가을, `4`: 겨울 |
| ③ | 휴일 (`holiday`) | 휴일 여부 | `1`: 휴일, `0`: 비 휴일 |
| ④ | 평일 (`workingday`) | 평일/주말 구분 | `1`: 평일, `0`: 주말 |
| ⑤ | 날씨 (`weather`) | 4단계 기상 상태 | `1`: 맑음 ~ `4`: 많은 눈/비 |
| ⑥ | 기온 (`temp`) | 측정 기온 (°C) | 최솟값 1, 최댓값 41 |
| ⑦ | 체감온도 (`atemp`) | 체감 온도 (°C) | 최솟값 1, 최댓값 45 |
| ⑧ | 습도 (`humidity`) | 상대 습도 (%) | 최솟값 0, 최댓값 100 |
| ⑨ | 풍속 (`windspeed`) | 측정 풍속 | 최솟값 0, 최댓값 57 |
| ⑩ | 비회원 대여 (`casual`) | 비회원 대여 수량 | 최솟값 0, 최댓값 367 |
| ⑪ | 회원 대여 (`registered`) | 회원 대여 수량 | 최솟값 0, 최댓값 886 |
| ⑫ | 총 대여 (`count`) | 총 자전거 대여 수량 | 최솟값 1, 최댓값 977 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 10,886행 × 12열 CSV 표.
- **시각화 데이터**: 계절, 휴일, 평일, 날씨 분포 막대그래프 제공.

---

## 5. 데이터셋 활용

1. **속성 전처리(정규화)** 및 **학습/테스트 분류(7:3, 8:2)**.
2. **상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 날씨 ↔ 총 대여 상관관계 및 역전 비교
   - Q2: 체감온도 ↔ 회원 대여 상관관계 및 역전 비교
   - Q3: 계절·평일·등급 집단별 총 대여량 분포 비교

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.033)의 Google Drive 링크(`1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD`)는 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.csv`를 내려받아 사용하세요. 또한 원문 코드의 `%%` 주석과 한글 인코딩 문제를 정리했습니다.

```python
# [T04] 자전거 대여량 예측 데이터셋 (Bike Sharing Demand) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.032-035
# 주의: 원문 Google Drive 링크는 만료(404)되었으므로 Kaggle에서 train.csv를 직접 내려받아 사용하세요.

import csv
import urllib.request
import numpy as np
import pandas as pd


def main():
    # 1. 데이터 다운로드 (원문 Google Drive 링크 만료로 주석 처리)
    # url = "https://drive.google.com/u/3/uc?id=1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD&export=download"
    # urllib.request.urlretrieve(url, "./bike.csv")
    # 대체 공식 링크: https://www.kaggle.com/c/bike-sharing-demand

    # 2. 내려받은 CSV 읽어오기
    cols = [
        "datetime", "season", "holiday", "workingday", "weather",
        "temp", "atemp", "humidity", "windspeed", "casual", "registered", "count",
    ]
    df = pd.read_csv("./bike.csv", names=cols, encoding="UTF-8")

    # 3. 날짜/시간 문자열에서 시간(00~23)만 추출 및 기온·체감온도·풍속 반올림 정수화
    for i in range(df.shape[0] - 1):
        for j in range(24):
            before = f" {j:02d}:00:00"
            after = f"{j}"
            if before in str(df["datetime"].iloc[i + 1]):
                df.loc[df.index[i + 1], "datetime"] = after

        df.loc[df.index[i + 1], "temp"] = int(np.round(float(df["temp"].iloc[i + 1])))
        df.loc[df.index[i + 1], "atemp"] = int(np.round(float(df["atemp"].iloc[i + 1])))
        df.loc[df.index[i + 1], "windspeed"] = int(np.round(float(df["windspeed"].iloc[i + 1])))

    # 4. 속성 명 한글 변환
    kr_names = {
        "datetime": "시간", "season": "계절", "holiday": "휴일", "workingday": "평일",
        "weather": "날씨", "temp": "기온", "atemp": "체감온도", "humidity": "습도",
        "windspeed": "풍속", "casual": "비회원 대여", "registered": "회원 대여", "count": "총 대여",
    }
    for k, v in kr_names.items():
        df.loc[0, k] = v

    # 5. CP949 저장 및 KNIME 호환 재저장
    df.to_csv("filename.csv", mode="w", index=False, header=False, encoding="CP949")

    with open("filename.csv", "r", encoding="CP949") as f:
        rdr = csv.reader(f)
        with open("bike.csv", "w", newline="", encoding="CP949") as f2:
            wr = csv.writer(f2)
            for line in rdr:
                wr.writerow(line)


if __name__ == "__main__":
    main()
```
