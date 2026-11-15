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
