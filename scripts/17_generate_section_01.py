from pathlib import Path

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
BASE = ROOT / "datasets" / "01_tabular_and_text"
BASE.mkdir(parents=True, exist_ok=True)

ITEMS = [
    {
        "slug": "01_insurance_cost",
        "num": "01",
        "title_kr": "보험료 예측 데이터셋",
        "title_en": "Medical Cost Personal Datasets",
        "pages": "p.008–014",
        "difficulty": "★☆☆☆☆ (1/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "회귀분석(Regression) / 지도학습",
        "url": "https://www.kaggle.com/datasets/mirichoi0218/insurance",
        "url_pdf": "https://www.kaggle.com/mirichoi0218/insurance",
        "url_status": "유효 (200 OK, Kaggle 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.70 (정규화), p.90 (학습데이터 시각화)",
        "summary": "대상자의 보험료 예측을 위한 데이터셋으로, 5가지 입력 요소(나이, 성별, BMI, 자녀 수, 흡연)와 보험료 간의 상관관계를 분석하는 회귀분석 입문용 데이터셋입니다.",
        "features": """- 정수로 나타내는 **나이(Age)**, 이분적 속성 **성별(Sex)**·**흡연(Smoker)**, 정량적 속성 **체질량지수(BMI)**·**자녀 수(Children)**·**보험료(Charges)** 등 총 **6개 속성**, **1,338개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 접근성 향상을 위해 원본의 `region` 속성을 제거하고, 달러(`$`) 단위 보험료를 원화(`₩`) 단위로 변환(100원 단위 버림)하는 가공이 포함되어 있습니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 나이 (`Age`) | 대상자의 나이 | 최솟값 18, 최댓값 64 |
| ② | 성별 (`Sex`) | 대상자의 성별 | `male` → `1`, `female` → `0` |
| ③ | 체질량지수 (`BMI`) | 키와 몸무게 기반 비만도 지수 | 최솟값 16, 최댓값 53.1 |
| ④ | 자녀 수 (`Children`) | 대상자의 자녀 수 | 최솟값 0, 최댓값 5 |
| ⑤ | 흡연 (`Smoker`) | 대상자의 흡연 여부 | `no` → `0`, `yes` → `1` |
| ⑥ | 보험료 (`Charges`) | 측정된 의료 보험료(원화 변환) | 최솟값 134,000원, 최댓값 7,652,000원 |""",
        "example": """- **원 데이터(Raw Data)**: 1,338행 × 6열(원본은 `region` 포함 7열)의 CSV 표 형태이며, 각 행이 한 대상자를 나타냅니다.
- **시각화 데이터**: 각 열별 막대/분포 히스토그램을 통해 나이(18~64), 성별 비율(male 51%, female 49%), BMI, 자녀 수(0~5) 분포를 직관적으로 확인할 수 있습니다.""",
        "utilization": """1. **데이터셋 정규화(Normalization)**: 단위와 값의 범위가 다른 속성들에서 평균을 빼고 표준편차로 나누어 평균 0, 표준편차 1 근처로 변환 (`인공지능과 미래사회 p.70`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 비율로 분할해 학습에 쓰지 않은 테스트 데이터로 성능을 평가.
3. **학습데이터 시각화**: 막대그래프로 속성별 분포 확인 (`인공지능과 미래사회 p.90`).
4. **각 요소 간 상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 나이 ↔ BMI 상관관계 및 입출력 역전 비교
   - Q2: 자녀 수 ↔ 보험료 상관관계 및 입출력 역전 비교
   - Q3: 성별·흡연 여부 집단 간 보험료 분포 차이 분석""",
        "code_note": "원문 코드(p.012~014)에서 `insurance['sex'][df['sex'] == ...]`로 미정의 변수 `df`를 참조한 오타와 MATLAB 스타일 `%%` 주석을 실행 가능한 Python 문법(`insurance` 변수 통일, `#` 주석)으로 정리했습니다. 코드 내 GitHub raw CSV URL은 현재도 정상 작동(200 OK)합니다.",
        "code": """# [T01] 보험료 예측 데이터셋 (Medical Cost Personal Datasets) 전처리 예제
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
""",
    },
    {
        "slug": "02_student_math_performance",
        "num": "02",
        "title_kr": "학생 수학 성적 예측 데이터셋",
        "title_en": "Student Performance Dataset",
        "pages": "p.015–021",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "회귀분석(Regression) / 지도학습",
        "url": "https://archive.ics.uci.edu/dataset/320/student+performance",
        "url_pdf": "https://archive.ics.uci.edu/ml/datasets/Student+Performance",
        "url_status": "유효 (200 OK, UCI 신규 경로로 리다이렉트)",
        "textbook": "없음",
        "summary": "학생들의 성별, 나이, 통학 시간, 공부 시간 등 8가지 속성과 수학 성적 간의 상관관계를 분석하는 회귀분석 데이터셋으로, 학생들의 실생활과 맞닿아 흥미도가 높습니다.",
        "features": """- 성별, 나이, 통학, 공부 시간, 이성 친구, 방과 후 자유시간, 건강, 결석, 수학 성적 등 **9개 속성**, **349개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: UCI 원본 데이터는 학교(`school`), 부모 교육수준, 1·2차 성적(`G1`, `G2`) 등을 포함해 총 33개 속성으로 이루어져 있으며, 포르투갈어 성적 데이터셋도 함께 제공됩니다. 본 자료집에서는 입문자 실습을 위해 9개 핵심 속성만 추출했습니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 성별 (`sex`) | 대상자의 성별 | 남자 `1`, 여자 `0` |
| ② | 나이 (`age`) | 대상자의 나이 | 최솟값 15, 최댓값 22 |
| ③ | 통학 (`traveltime`) | 집에서 학교까지 이동 시간 | `1`: 15분 미만, `2`: 15~30분, `3`: 30~60분, `4`: 60분 이상 |
| ④ | 공부 시간 (`studytime`) | 주당 공부 소요 시간 | `1`: 2시간 이하, `2`: 2~5시간, `3`: 5~10시간, `4`: 10시간 이상 |
| ⑤ | 이성 친구 (`romantic`) | 이성 친구 유무 | 있음 `1`, 없음 `0` |
| ⑥ | 방과 후 자유시간 (`freetime`) | 설문 기반 자유시간 수준 | `1`(매우 적음) ~ `5`(매우 많음) |
| ⑦ | 건강 (`health`) | 설문 기반 현재 건강 상태 | `1`(매우 나쁨) ~ `5`(매우 건강) |
| ⑧ | 결석 (`absences`) | 총 결석 일수 | 최솟값 0, 최댓값 93 |
| ⑨ | 수학 성적 (`G3`) | 3차 시험 기준 최종 수학 성적 | 최솟값 3, 최댓값 19 |""",
        "example": """- **원 데이터(Raw Data)**: 349행 × 9열의 가공 CSV 형태(성별, 나이, 통학시간, 공부시간, 이성친구, 방과후자유시간, 건강, 결석, 수학성적).
- **시각화 데이터**: 나이, 통학 시간, 공부 시간 등 각 열의 빈도 분포 막대그래프 제공.""",
        "utilization": """1. **학습/테스트 데이터 분류**: 7:3 또는 8:2 비율 분할.
2. **각 속성 간 상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 공부 시간(입력) ↔ 수학 성적(출력) 상관관계 및 입출력 역전 비교
   - Q2: 통학 시간(입력) ↔ 수학 성적(출력) 상관관계 및 입출력 역전 비교
   - Q3: 이성 친구 유무·건강 상태 집단별 수학 성적 분포 차이 분석""",
        "code_note": "원문 코드(p.019)에 수록된 개인 Google Drive 다운로드 링크(`1ROipU8RCoiw9gXl-78jOTT4BfjGyhg9Y`)는 현재 만료(HTTP 404)되었습니다. UCI 공식 페이지에서 `student.zip`을 내려받아 `student-mat.csv`(세미콜론 구분자 `;`)를 사용하시기 바랍니다.",
        "code": """# [T02] 학생 수학 성적 예측 데이터셋 (Student Performance) 전처리 예제
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
""",
    },
    {
        "slug": "03_wine_quality",
        "num": "03",
        "title_kr": "포도주 품질 예측 데이터셋",
        "title_en": "Red Wine Quality",
        "pages": "p.022–028",
        "difficulty": "★☆☆☆☆ (1/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "회귀분석(Regression) / 지도학습",
        "url": "https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009",
        "url_pdf": "https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009",
        "url_status": "유효 (200 OK, Kaggle 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.70 (정규화)",
        "summary": "적포도주의 화학적 성분(구연산, 잔류 당분, 염화나트륨, 밀도, pH, 황산칼륨, 도수)과 포도주 품질 점수 간의 상관관계를 다루는 회귀분석(및 분류 변환 가능) 데이터셋입니다.",
        "features": """- 7가지 이화학적 속성 + 품질(`quality`) 등 총 **8개 속성**, **1,599개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 원본 데이터는 고정산도(`fixed acidity`), 휘발성 산도(`volatile acidity`) 등을 포함한 총 12개 속성이며, 실습 편의를 위해 4개 속성을 제외하고 8개 속성만 사용합니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 |
|---|---|---|---|
| ① | 구연산 농도 (`citric acid`) | 포도주의 구연산 농도 | 최솟값 0, 최댓값 1 |
| ② | 잔류 당분 농도 (`residual sugar`) | 발효 후 남은 당분 농도 | 최솟값 0.9, 최댓값 15.5 |
| ③ | 염화나트륨 농도 (`chlorides`) | 소금 성분 농도 | 최솟값 0.012, 최댓값 0.611 |
| ④ | 밀도 (`density`) | 알코올 도수와 당분 함량의 영향을 받는 밀도 | 최솟값 0.990, 최댓값 1.003 |
| ⑤ | pH | 수소 이온 농도 지수(산성/알칼리성) | 최솟값 2.74, 최댓값 4.01 |
| ⑥ | 황산칼륨 농도 (`sulphates`) | 황산칼륨(방부/산화방지) 농도 | 최솟값 0.33, 최댓값 2 |
| ⑦ | 도수 (`alcohol`) | 알코올 도수 | 최솟값 8.4, 최댓값 14.9 |
| ⑧ | 품질 (`quality`) | 포도주 품질 점수 | 0(나쁨) ~ 10(좋음) |""",
        "example": """- **원 데이터(Raw Data)**: 1,599행 × 8열(구연산, 잔류 당분, 염화나트륨, 밀도, pH, 황산칼륨, 도수, 품질)의 CSV 표.
- **시각화 데이터**: Kaggle 시각화 기준 각 성분별 히스토그램 제공.""",
        "utilization": """1. **데이터셋 정규화(Normalization)** (`인공지능과 미래사회 p.70`)
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 도수(입력) ↔ 밀도(출력) 상관관계 및 역전 비교
   - Q2: 잔류 당분(입력) ↔ 품질(출력) 상관관계 및 역전 비교
   - Q3: 계절·평일 여부 등에 따른 비교(원문 Q3 문구 일부는 T04와 혼재)
4. **회귀분석 문제를 분류 문제로 변환해보기**: 품질 0~5점을 `0`(나쁨), 6점 이상을 `1`(좋음)으로 이진화하거나 3단계로 범주화해 분류 모델 실험.""",
        "code_note": "원문 코드(p.026)의 Google Drive 링크(`18w9sj9bwY7nU1z51nqCg3bfAcAU7C-qXA`)는 현재 만료(404)되었으며, Kaggle 공식 링크에서 `winequality-red.csv`를 내려받아 사용할 수 있습니다.",
        "code": """# [T03] 포도주 품질 예측 데이터셋 (Red Wine Quality) 전처리 예제
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
""",
    },
    {
        "slug": "04_bike_sharing_demand",
        "num": "04",
        "title_kr": "자전거 대여량 예측 데이터셋",
        "title_en": "Bike Sharing Demand",
        "pages": "p.029–035",
        "difficulty": "★☆☆☆☆ (1/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "회귀분석(Regression) / 지도학습",
        "url": "https://www.kaggle.com/c/bike-sharing-demand",
        "url_pdf": "https://www.kaggle.com/c/bike-sharing-demand",
        "url_status": "유효 (200 OK, Kaggle Competition 공식 페이지)",
        "textbook": "없음",
        "summary": "날씨, 시간, 휴일 등 12가지 속성과 자전거 대여량(비회원/회원/총 대여) 간의 상관관계를 분석하는 회귀분석 데이터셋입니다.",
        "features": """- 시간, 계절, 휴일, 평일, 날씨, 기온, 체감온도, 습도, 풍속, 비회원 대여, 회원 대여, 총 대여 등 **12개 속성**, **10,886개 데이터 포인트**로 구성됩니다.
- **원본 데이터와의 차이**: 날짜+시간 문자열(`datetime`)에서 날짜를 제거하고 시간(`0~23`)만 추출했으며, 기온·체감온도·풍속을 반올림해 정수로 변환했습니다.
- **편집 참고**: p.029 도입문 첫 문장에 "총 9가지의 속성"이라는 오기(실제는 12개 속성)가 섞여 있습니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
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
| ⑫ | 총 대여 (`count`) | 총 자전거 대여 수량 | 최솟값 1, 최댓값 977 |""",
        "example": """- **원 데이터(Raw Data)**: 10,886행 × 12열 CSV 표.
- **시각화 데이터**: 계절, 휴일, 평일, 날씨 분포 막대그래프 제공.""",
        "utilization": """1. **속성 전처리(정규화)** 및 **학습/테스트 분류(7:3, 8:2)**.
2. **상관관계 파악 탐구 (Q1~Q3)**:
   - Q1: 날씨 ↔ 총 대여 상관관계 및 역전 비교
   - Q2: 체감온도 ↔ 회원 대여 상관관계 및 역전 비교
   - Q3: 계절·평일·등급 집단별 총 대여량 분포 비교""",
        "code_note": "원문 코드(p.033)의 Google Drive 링크(`1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD`)는 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.csv`를 내려받아 사용하세요. 또한 원문 코드의 `%%` 주석과 한글 인코딩 문제를 정리했습니다.",
        "code": """# [T04] 자전거 대여량 예측 데이터셋 (Bike Sharing Demand) 전처리 예제
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
""",
    },
    {
        "slug": "05_titanic_survival",
        "num": "05",
        "title_kr": "타이타닉 생존자 예측 데이터셋",
        "title_en": "Titanic - Machine Learning from Disaster",
        "pages": "p.036–042",
        "difficulty": "★☆☆☆☆ (1/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "분류(Classification - 이진 분류) / 지도학습",
        "url": "https://www.kaggle.com/c/titanic",
        "url_pdf": "https://www.kaggle.com/c/titanic",
        "url_status": "유효 (200 OK, Kaggle Competition 공식 페이지)",
        "textbook": "없음",
        "summary": "타이타닉 호 탑승자의 입장권 등급, 성별, 나이, 가족 수, 요금, 승선 항구 정보를 바탕으로 생존 여부를 예측하는 대표적 이진 분류 입문 데이터셋입니다.",
        "features": """- 총 **8개 속성**으로 구성되며, 나이 결측치 제거 등의 정제 과정을 거쳐 자료집 기준 **434개**(예시 표 설명에는 714행 언급)의 데이터 포인트를 활용합니다.
- **원본 데이터와의 차이**: 원본 10개(PassengerId, Name, Ticket, Cabin 포함)에서 식별자·문자열 4개 속성을 제거하고, 운임(`Fare`)을 원화 단위로 변환, 나이(`Age`)를 반올림 정수화하며 생존 여부(`Survived`) 열을 맨 뒤로 재배치했습니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 인코딩 |
|---|---|---|---|
| ① | 입장권 등급 (`Pclass`) | 승차권 등급 | `1`: 1등급, `2`: 2등급, `3`: 3등급 |
| ② | 성별 (`Sex`) | 대상자 성별 | `1`: 남자, `0`: 여자 |
| ③ | 나이 (`Age`) | 대상자 나이(반올림 정수) | 최솟값 0, 최댓값 81 |
| ④ | 탑승 가족(형제) (`SibSp`) | 함께 탄 형제/배우자 수 | 최솟값 0, 최댓값 5 |
| ⑤ | 탑승 가족(부모, 자식) (`Parch`) | 함께 탄 부모/자식 수 | 최솟값 0, 최댓값 6 |
| ⑥ | 입장권 요금 (`Fare`) | 원화 변환 탑승권 요금 | 최솟값 0, 최댓값 614,000 |
| ⑦ | 승선 항구 (`Embarked`) | 탑승 항구 | `0`: S(사우스햄튼), `1`: C(셰르부르), `2`: Q(퀸즈타운) |
| ⑧ | 생존 여부 (`Survived`) | 생존 여부(목표 변수) | `1`: 생존, `0`: 사망 |""",
        "example": """- **원 데이터(Raw Data)**: 결측치(`Age`) 제거 후 구성된 CSV 표.
- **시각화 데이터**: Pclass, Sex, Age, SibSp 분포 시각화 및 등급별 생존자/사망자 막대그래프 제공.""",
        "utilization": """1. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
2. **생존 여부 분류 탐구 (Q1~Q3)**:
   - Q1: 나이 입력 vs 다중 속성 입력 성능 비교
   - Q2: 입장권 등급 입력 vs 다중 속성 조합 성능 비교
   - Q3: 입장권 등급, 나이 등 집단별 생존율 분포 차이 분석""",
        "code_note": "원문 코드(p.039)의 `blog.kakaocdn.net` 링크는 현재 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.csv`를 내려받아 사용하세요.",
        "code": """# [T05] 타이타닉 생존자 예측 데이터셋 (Titanic) 전처리 예제
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
""",
    },
    {
        "slug": "06_spam_sms",
        "num": "06",
        "title_kr": "스팸 SMS 분류 데이터셋",
        "title_en": "SMS Spam Collection",
        "pages": "p.043–048",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "자연어 처리 / 이진 분류(Classification) / 지도학습",
        "url": "https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
        "url_pdf": "https://archive.ics.uci.edu/ml/datasets/sms+spam+collection",
        "url_status": "유효 (200 OK, UCI 신규 경로 리다이렉트)",
        "textbook": "인공지능과 미래사회 p.236 (텍스트 전처리), p.240 (순환 신경망 분류)",
        "summary": "영문 SMS 문자가 정상(ham)인지 스팸(spam)인지 판별하는 이진 텍스트 분류 입문용 데이터셋입니다.",
        "features": """- SMS 메시지와 스팸 여부 2개 속성, 총 **5,572개** 메시지(정상 87%, 스팸 13%)로 구성되며, 중복 제거 시 5,169개입니다.
- **텍스트 전처리 핵심**: 토큰화(Tokenization), 정규화(Normalization), 정수 인코딩 및 패딩(`pad_sequences`, `maxlen=100`) 과정을 학습합니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 예시 |
|---|---|---|---|
| ① | 스팸 여부 (`v1`) | 정상/스팸 라벨 | `ham`: 일반 SMS, `spam`: 스팸 SMS |
| ② | SMS (`v2`) | 영문 문자 내용 | 예: `"Oh k...i'm watching here:)"` |""",
        "example": """- **원 데이터(Raw Data)**: 5,572개 행 중 중복 제거된 5,169개 데이터.
- **전처리 데이터**: 정수 인덱스로 치환되고 `maxlen=100`으로 제로 패딩된 100차원 숫자 시퀀스 표.""",
        "utilization": """1. **텍스트 데이터 전처리 과정 이해하기**: 원-핫 인코딩 및 정수 인코딩 비교 실습 (`인공지능과 미래사회 p.236`).
2. **순환 신경망(RNN)으로 텍스트 데이터 분류해보기**: 스팸 SMS 판별 모델 설계 (`인공지능과 미래사회 p.240`).""",
        "code_note": "코드 내 예시 GitHub raw URL(`mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv`)은 현재도 정상 작동(200 OK)합니다.",
        "code": """# [T06] 스팸 SMS 분류 데이터셋 (SMS Spam Collection) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.046-048

import urllib.request
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def main():
    # 1. 데이터 다운로드 (GitHub 미러 - 200 OK 확인됨)
    url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
    urllib.request.urlretrieve(url, "./spam.csv")

    # 2. CSV 로드 및 중복 제거
    df = pd.read_csv("./spam.csv", encoding="latin1")
    df.drop_duplicates(subset=["v2"], inplace=True)

    sms_txts = df["v2"]
    labels = df["v1"].map({"ham": 0, "spam": 1}).values

    # 3. 토큰화 및 정수 시퀀스 변환
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sms_txts)
    sequences = tokenizer.texts_to_sequences(sms_txts)

    # 4. 패딩 (max_len = 100)
    max_len = 100
    padded_txts = pad_sequences(sequences, maxlen=max_len)

    # 5. 학습/테스트 분할 (8:2) 및 라벨 병합 저장
    n_of_train = int(len(sequences) * 0.8)
    sms_train = padded_txts[:n_of_train]
    label_train = np.array(labels[:n_of_train])
    sms_test = padded_txts[n_of_train:]
    label_test = np.array(labels[n_of_train:])

    trains = np.hstack((sms_train, label_train.reshape(-1, 1)))
    tests = np.hstack((sms_test, label_test.reshape(-1, 1)))

    pd.DataFrame(trains).to_csv("spam_train.csv", index=False)
    pd.DataFrame(tests).to_csv("spam_test.csv", index=False)


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "07_korean_hate_speech",
        "num": "07",
        "title_kr": "한국어 혐오 분류 데이터셋",
        "title_en": "Korean Extremist Website Womad Hate Speech Data",
        "pages": "p.049–054",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "자연어 처리 / 이진 분류(Classification) / 지도학습",
        "url": "https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data",
        "url_pdf": "https://www.kaggle.com/captainnemo9292/korean-extremist-website-womad-hate-speech-data",
        "url_status": "유효 (200 OK, Kaggle 공식 페이지 브라우저 검증 완료)",
        "textbook": "인공지능과 미래사회 p.236 (텍스트 전처리)",
        "summary": "한국어 문장에 혐오 표현이 포함되어 있는지 여부를 분류하는 총 2,000문장 규모의 한국어 NLP 입문 데이터셋입니다.",
        "features": """- 혐오 발언 여부와 한국어 문장 2개 속성, 총 **2,000개 문장**(혐오 발언 50%, 일반 문장 50%)으로 구성됩니다.
- **한국어 NLP 특성**: 영어와 달리 어순보다 조사의 역할이 크고, 띄어쓰기 오류·주어 생략·억양 의존 문장이 많아 형태소 분석(`KoNLPy Okt`) 전처리가 필수적입니다.
- **편집 참고**: p.049에 `데이터셋 목적` 라벨이 2번 중복 인쇄되어 있으며, 두 번째 항목은 실제로 `데이터셋 특징` 내용입니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 범위 / 값 |
|---|---|---|---|
| ① | 혐오 발언 여부 | 혐오 발언 여부 라벨 | `0`: 일반, `1`: 혐오 발언 |
| ② | 문장 | 한국어 텍스트 문장 | 문자열 (총 2,000건) |""",
        "example": """- **원 데이터(Raw Data)**: 2,000개 문장과 라벨(0/1) 표.
- **전처리 데이터**: 형태소(`Okt`) 토큰화 후 빈도 상위 4,885개 단어 기준 Bag-of-Words(BoW) 이진/빈도 벡터 표.""",
        "utilization": """1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **한국어 혐오 분류를 위한 신경망 모델 설계하기**.
3. **학습데이터 추가하기**: 엉터리 데이터 등 새 데이터를 추가해 모델 성능 변화와 데이터 품질의 중요성 탐구.""",
        "code_note": "Kaggle 데이터셋 페이지는 로그인 또는 Kaggle CLI(`kaggle datasets download -d captainnemo9292/korean-extremist-website-womad-hate-speech-data`)를 통해 `hate_speech_data.csv`를 내려받아야 합니다(단순 `urllib` 호출은 불가). 실행 시 `konlpy`, `nltk` 설치가 필요합니다.",
        "code": """# [T07] 한국어 혐오 분류 데이터셋 전처리 예제 (BoW 벡터화)
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.052-054
# 주의: Kaggle 데이터셋은 직접 URL 다운로드 대신 Kaggle 웹 또는 Kaggle CLI로 내려받아야 합니다.
#       kaggle datasets download -d captainnemo9292/korean-extremist-website-womad-hate-speech-data

import nltk
import numpy as np
import pandas as pd
from konlpy.tag import Okt


def main():
    okt = Okt()

    def tokenize(doc):
        temp = []
        for t in okt.pos(doc, norm=True, stem=True):
            if t[1] != "Josa" and t[1] != "Punctuation":
                temp.append("/".join(t))
        return temp

    # 1. 내려받은 데이터 로드 및 불필요 열 제거
    df = pd.read_csv("hate_speech_data.csv", encoding="UTF-8")
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)

    txts = df["문장"]
    labels = df["혐오 여부"]

    # 2. 품사 단위 토큰화 및 전체 토큰 수집 (상위 4,885개)
    tokenize_txts = [tokenize(sentence) for sentence in txts]
    tokens = [t for d in tokenize_txts for t in d]
    text = nltk.Text(tokens, name="NMSC")
    selected_words = [f[0] for f in text.vocab().most_common(4885)]

    def bag_of_words(doc):
        return [doc.count(word) for word in selected_words]

    # 3. BoW 수치화 및 train/test 분할 (1,600 : 400)
    vectorized_txts = np.asarray([bag_of_words(d) for d in tokenize_txts])
    labels_arr = np.asarray(labels)

    train_txts = vectorized_txts[:1600]
    test_txts = vectorized_txts[1600:]
    train_labels = labels_arr[:1600]
    test_labels = labels_arr[1600:]

    trains = np.hstack((train_txts, train_labels.reshape(-1, 1)))
    tests = np.hstack((test_txts, test_labels.reshape(-1, 1)))

    pd.DataFrame(trains).to_csv("hate_train.csv", index=False)
    pd.DataFrame(tests).to_csv("hate_test.csv", index=False)


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "08_naver_movie_review_nsmc",
        "num": "08",
        "title_kr": "네이버 영화 리뷰데이터(NSMC)",
        "title_en": "Naver Sentiment Movie Corpus (NSMC)",
        "pages": "p.055–060",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "TXT 파일",
        "category": "수치/텍스트",
        "purpose": "자연어 처리 / 이진 분류(감성 분석) / 지도학습",
        "url": "https://github.com/e9t/nsmc",
        "url_pdf": "https://github.com/e9t/nsmc.git",
        "url_status": "유효 (200 OK, GitHub 공식 저장소)",
        "textbook": "없음",
        "summary": "네이버 영화 리뷰 텍스트를 긍정(1) 또는 부정(0)으로 분류한 총 20만 건의 대규모 한국어 감성 분석 표준 벤치마크 데이터셋입니다.",
        "features": """- 리뷰 고유 key(`id`), 리뷰 텍스트(`document`), 감성 라벨(`label`) 3개 속성으로 구성됩니다.
- **데이터 규모**: `train` 150,000개, `test` 50,000개 등 총 **200,000개** 리뷰를 포함합니다(전체 등장 토큰 약 534,534개, 예제 코드에서는 빈도 상위 5,000개 선택).""",
        "structure": """| 번호 | 속성명 | 설명 | 예시 / 값 |
|---|---|---|---|
| ① | `id` | 리뷰 고유 식별 정수 | 예: `9976970` |
| ② | `document` | 한국어 영화 리뷰 문장 | 예: `"청춘 영화의 최고봉..."` |
| ③ | `label` | 긍정/부정 감성 라벨 | `0`: 부정, `1`: 긍정 |""",
        "example": """- **원 데이터(Raw Data)**: 탭(`\t`)으로 구분된 150,000행 × 3열(`ratings_train.txt`) 및 50,000행 × 3열(`ratings_test.txt`).""",
        "utilization": """1. **한글 텍스트 데이터셋 전처리(정규화)하기**:
   - Q1-1 / Q1-2: 글자 단위 vs 단어(형태소) 단위 토큰화 성능 및 실행 시간 비교.
2. **텍스트의 감정 분류하기**:
   - Q2-1: `vocab_size` 변화에 따른 성능 변화 분석.
   - Q2-2: 이진 분류 임곗값(threshold) 변화 실험.""",
        "code_note": "원문 코드(p.057~058) 상단에 BERT 관련 import(`transformers`, `torch`, `keras`, `sklearn`)와 하단에 `Okt` + BoW 전처리 코드가 함께 수록되어 있습니다. 쉘 명령(`!pip install`, `!git clone`)은 주석으로 보존했습니다.",
        "code": """# [T08] 네이버 영화 리뷰 데이터셋 (NSMC) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.057-060
# [Colab/Shell] !pip install transformers konlpy
# [Colab/Shell] !git clone https://github.com/e9t/nsmc.git

import os
import nltk
import numpy as np
import pandas as pd
from konlpy.tag import Okt

# 원문 p.057 상단에 소개된 BERT/PyTorch 활용 시 필요한 주요 라이브러리 참고:
# import torch
# from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
# from transformers import get_linear_schedule_with_warmup
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from keras.preprocessing.sequence import pad_sequences
# from sklearn.model_selection import train_test_split


def main():
    okt = Okt()

    def tokenize(doc):
        temp = []
        for t in okt.pos(doc, norm=True, stem=True):
            if t[1] != "Josa" and t[1] != "Punctuation":
                temp.append("/".join(t))
        return temp

    # 1. NSMC train/test 로드 (탭 구분 txt)
    train = pd.read_csv("nsmc/ratings_train.txt", sep="\\t")
    test = pd.read_csv("nsmc/ratings_test.txt", sep="\\t")

    train_comments = [str(s) for s in train["document"]]
    train_labels = [int(g) for g in train["label"]]

    # 2. 품사 단위 토큰화 및 빈도 상위 vocab_size=5000 선택
    vocab_size = 5000
    tokenize_comments = [tokenize(comment) for comment in train_comments]
    tokens = [t for d in tokenize_comments for t in d]
    text = nltk.Text(tokens, name="NMSC")
    selected_words = [f[0] for f in text.vocab().most_common(vocab_size)]

    def bag_of_words(doc):
        return [doc.count(word) for word in selected_words]

    # 3. BoW 수치화 및 라벨 병합 (150,000 x 5,001)
    vec_comments = np.asarray([bag_of_words(d) for d in tokenize_comments])
    vec_labels = np.asarray(train_labels)
    trains = np.hstack((vec_comments, vec_labels.reshape(-1, 1)))

    # 4. CSV 저장
    df_train = pd.DataFrame(trains)
    train_dir = "./train"
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    df_train.to_csv(os.path.join(train_dir, "movie_review_train.csv"), index=False)


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "09_korean_intent_3i4k",
        "num": "09",
        "title_kr": "한국어 의도 분류 데이터셋",
        "title_en": "3i4k (Intention Identification in Korean)",
        "pages": "p.061–066",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "TXT 파일",
        "category": "수치/텍스트",
        "purpose": "자연어 처리 / 다중 분류(7개 발화 의도) / 지도학습",
        "url": "https://github.com/warnikchow/3i4k",
        "url_pdf": "https://github.com/warnikchow/3i4k",
        "url_status": "유효 (200 OK, GitHub 공식 저장소)",
        "textbook": "인공지능과 미래사회 p.236 (텍스트 전처리)",
        "summary": "한국어 발화가 가진 7가지 의도(불완전 발화, 서술, 질문, 요구, 수사 의문, 수사 명령, 억양 의존)를 분류하는 55,134문장 규모의 다중 분류 데이터셋입니다.",
        "features": """- 한국어 문장과 0~6까지의 의도 라벨 2개 속성, 총 **55,134개 문장**으로 구성됩니다.
- **편집 참고**: p.061 도입문과 p.063 예시 설명에 "전체 2,000개의 문장"(직전 T07 한국어 혐오 분류 데이터셋 문구 잔재)이라는 오기가 있으나, 실제 데이터셋 특징 및 원 논문/GitHub 기준은 55,134개(50,000건 이상) 문장입니다.""",
        "structure": """| 라벨 | 의도 분류 | 설명 |
|---|---|---|
| `0` | Fragments | 불완전하게 끝난 문장 / 조각 발화 |
| `1` | 서술 (Statement) | 다른 분류에 속하지 않는 일반 평서문 |
| `2` | 질문 (Question) | 상대에게 답변을 요구하는 의문문 |
| `3` | 요구 (Command) | 상대에게 행위를 요구하는 명령/청유문 |
| `4` | 수사 의문 (Rhetorical Question) | 의문문 형태이나 답변을 요구하지 않는 문장 |
| `5` | 수사 명령 (Rhetorical Command) | 명령형 형태이나 행위를 강요하지 않는 문장 (예: "즐거운 하루 보내세요") |
| `6` | 억양 의존 (Intonation-dependent) | 억양에 따라 의도가 달라지는 발화 |""",
        "example": """- **원 데이터(Raw Data)**: `라벨\\t문장` 형태의 탭 구분 TXT 파일 (`fsi.txt`).
- **전처리 데이터**: `Okt` 토큰화 후 상위 5,000개 토큰 기준 5,000차원 BoW + 라벨(총 5,001열) 벡터 표.""",
        "utilization": """1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **한국어 의도 분류를 위한 신경망 모델 설계하기** (※ 본문 중 '혐오 분류' 오기 존재).""",
        "code_note": "원문 코드(p.064~066)에서 `line.split('\\t')`의 탭 문자 표기가 좁아져 있으며, `kor_train.csv`/`kor_test.csv` 저장 과정을 Python 표준 문법에 맞게 정확히 복원했습니다.",
        "code": """# [T09] 한국어 의도 분류 데이터셋 (3i4k) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.064-066
# 데이터셋 출처: https://github.com/warnikchow/3i4k (fci_train_val.txt / fsi.txt 등)

import nltk
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split


def read_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = [line.split("\\t") for line in f.read().splitlines() if "\\t" in line]
    return data


def main():
    okt = Okt()

    def tokenize(doc):
        temp = []
        for t in okt.pos(doc, norm=True, stem=True):
            if t[1] != "Josa" and t[1] != "Punctuation":
                temp.append("/".join(t))
        return temp

    # 1. 텍스트 데이터 읽기
    trains = read_data("./fsi.txt")
    tokenize_txts = [tokenize(datas[1]) for datas in trains]
    labels = [datas[0] for datas in trains]

    # 2. 품사 단위 토큰 수집 및 상위 5,000개 단어 선택
    tokens = [t for d in tokenize_txts for t in d]
    text = nltk.Text(tokens, name="3i4k")
    selected_words = [f[0] for f in text.vocab().most_common(5000)]

    def bag_of_words(doc):
        return [doc.count(word) for word in selected_words]

    # 3. BoW 수치화 및 train/test(8:2) 분할
    vectorized_txts = np.asarray([bag_of_words(d) for d in tokenize_txts])
    labels_arr = np.asarray(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        vectorized_txts, labels_arr, test_size=0.2, random_state=42
    )

    train_arr = np.hstack((X_train, y_train.reshape(-1, 1)))
    test_arr = np.hstack((X_test, y_test.reshape(-1, 1)))

    pd.DataFrame(train_arr).to_csv("kor_train.csv", index=False)
    pd.DataFrame(test_arr).to_csv("kor_test.csv", index=False)


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "10_korean_chatbot",
        "num": "10",
        "title_kr": "한국어 챗봇 데이터셋",
        "title_en": "Chatbot_data (Korean Chatbot Q&A)",
        "pages": "p.067–073",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "CSV 파일",
        "category": "수치/텍스트",
        "purpose": "Question Answering, Text Generation / 지도학습",
        "url": "https://github.com/songys/Chatbot_data",
        "url_pdf": "https://github.com/songys/Chatbot_data",
        "url_status": "유효 (200 OK, GitHub 공식 저장소)",
        "textbook": "인공지능과 미래사회 p.236 (텍스트 전처리)",
        "summary": "인간의 질문(Q)과 챗봇의 대답(A) 11,876쌍 및 대화 상황 라벨(일상/이별/사랑)로 구성된 한국어 대화 생성·질의응답 데이터셋입니다.",
        "features": """- 질문(`Q`), 대답(`A`), 상황 라벨(`label`) 3개 열, 총 **11,876쌍**으로 구성됩니다.
- **전처리 포인트**: 질문과 대답의 전체 어휘를 합쳐 상위 2,000개 공통 토큰으로 사전을 만든 뒤, 질문 2,000차원 + 대답 2,000차원 = 총 4,000차원 벡터로 병합 저장하는 예제를 제시합니다.""",
        "structure": """| 번호 | 속성명 | 설명 | 값 / 범위 |
|---|---|---|---|
| ① | `Q` | 인간의 질문 문장 | 한국어 텍스트 |
| ② | `A` | 챗봇의 대답 문장 | 한국어 텍스트 |
| ③ | `label` | 대화 상황 카테고리 | `0`: 일상다반사, `1`: 이별(부정), `2`: 사랑(긍정) |""",
        "example": """- **원 데이터(Raw Data)**: 11,876행 × 3열 CSV (`Q`, `A`, `label`).
- **전처리 데이터**: 질문 2,000차원 + 대답 2,000차원 = 4,000열 벡터 CSV (`K_chatbot_train.csv`, `K_chatbot_test.csv`).""",
        "utilization": """1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **텍스트 인식과 대답 생성 모델에 대해 알아보기**: 단순 분류 모델이 아니라 연속적 문장 생성 모델(Seq2Seq 등)의 필요성과 구조 탐구.""",
        "code_note": "원문 코드(p.070~073)는 `Korpora` 패키지를 통해 데이터를 내려받으며, 쉘 명령(`!git clone`, `pip install Korpora`)을 주석으로 정리했습니다.",
        "code": """# [T10] 한국어 챗봇 데이터셋 (Chatbot_data) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.070-073
# [Colab/Shell] !git clone https://github.com/ko-nlp/Korpora
# [Colab/Shell] pip install Korpora konlpy

import nltk
import numpy as np
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt


def main():
    # 1. Korpora를 통한 데이터셋 다운로드
    Korpora.fetch("korean_chatbot_data")
    df = pd.read_csv("./Korpora/korean_chatbot_data/ChatbotData.csv", encoding="UTF-8")

    okt = Okt()

    def tokenize(doc):
        temp = []
        for t in okt.pos(doc, norm=True, stem=True):
            if t[1] != "Josa" and t[1] != "Punctuation":
                temp.append("/".join(t))
        return temp

    txts_Q = df["Q"]
    txts_A = df["A"]

    # 2. 질문(Q)과 대답(A) 각각 품사 토큰화 후 공통 어휘 2,000개 추출
    tokenize_txts_Q = [tokenize(sentence) for sentence in txts_Q]
    tokenize_txts_A = [tokenize(sentence) for sentence in txts_A]

    tokens_Q = [t for d in tokenize_txts_Q for t in d]
    tokens_A = [t for d in tokenize_txts_A for t in d]
    tokens_all = tokens_Q + tokens_A

    text_all = nltk.Text(tokens_all, name="Chatbot")
    selected_words_all = [f[0] for f in text_all.vocab().most_common(2000)]

    def bag_of_words(doc):
        return [doc.count(word) for word in selected_words_all]

    # 3. BoW 수치화
    vectorized_txts_Q = np.asarray([bag_of_words(d) for d in tokenize_txts_Q])
    vectorized_txts_A = np.asarray([bag_of_words(d) for d in tokenize_txts_A])

    # 4. 학습(300쌍)/테스트(100쌍) 분할 및 4,000차원 병합 저장
    train_num = 300
    test_num = 100

    train_Q = vectorized_txts_Q[:train_num]
    train_A = vectorized_txts_A[:train_num]
    test_Q = vectorized_txts_Q[train_num : train_num + test_num]
    test_A = vectorized_txts_A[train_num : train_num + test_num]

    trains = np.hstack((train_Q, train_A))
    tests = np.hstack((test_Q, test_A))

    pd.DataFrame(trains).to_csv("K_chatbot_train.csv", index=False)
    pd.DataFrame(tests).to_csv("K_chatbot_test.csv", index=False)


if __name__ == "__main__":
    main()
""",
    },
]


def render_readme(it: dict) -> str:
    return f"""# {it['num']}. {it['title_kr']} ({it['title_en']})

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), {it['pages']}

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | {it['title_kr']} ({it['title_en']}) |
| **카테고리** | {it['category']} |
| **데이터셋 목적** | {it['purpose']} |
| **난이도** | {it['difficulty']} |
| **흥미도** | {it['interest']} |
| **파일 형태** | {it['format']} |
| **데이터셋 링크** | {it['url']} |
| **원문 PDF 표기 링크** | {it['url_pdf']} |
| **링크 검증 상태 (2026-11)** | {it['url_status']} |
| **교과서 연계** | {it['textbook']} |

{it['summary']}

---

## 2. 데이터셋 특징

{it['features']}

---

## 3. 데이터셋 구성

{it['structure']}

---

## 4. 데이터셋 예시

{it['example']}

---

## 5. 데이터셋 활용

{it['utilization']}

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: {it['code_note']}

```python
{it['code'].strip()}
```
"""


def main() -> None:
    index_rows = []
    for it in ITEMS:
        d = BASE / it["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(render_readme(it), encoding="utf-8")
        (d / "prepare_dataset.py").write_text(it["code"], encoding="utf-8")
        index_rows.append(
            f"| {it['num']} | [{it['title_kr']}](./{it['slug']}/README.md) | {it['purpose']} | {it['difficulty']} | {it['interest']} | {it['format']} | [링크]({it['url']}) | [코드](./{it['slug']}/prepare_dataset.py) |"
        )

    section_readme = f"""# Ⅰ. 수치/텍스트 데이터셋 (Tabular & Text Datasets)

정형 수치 데이터(회귀·이진 분류 5종)와 자연어 텍스트 데이터(영문·한국어 분류 및 챗봇 5종) 총 10개 데이터셋을 수록합니다.

| 번호 | 데이터셋 명 | 데이터셋 목적 | 난이도 | 흥미도 | 형태 | 공식 링크 | 예시 코드 |
|---|---|---|---|---|---|---|---|
{chr(10).join(index_rows)}
"""
    (BASE / "README.md").write_text(section_readme, encoding="utf-8")
    print(f"Generated {len(ITEMS)} datasets in {BASE}")


if __name__ == "__main__":
    main()
