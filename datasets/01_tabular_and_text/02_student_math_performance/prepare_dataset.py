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
