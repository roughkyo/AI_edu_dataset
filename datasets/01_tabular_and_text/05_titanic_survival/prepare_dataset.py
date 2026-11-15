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
