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
