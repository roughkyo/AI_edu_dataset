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
