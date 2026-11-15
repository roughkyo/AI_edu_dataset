# [T06] 스팸 SMS 분류 데이터셋 (SMS Spam Collection) 전처리 예제
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
