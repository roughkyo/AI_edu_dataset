# [T09] 한국어 의도 분류 데이터셋 (3i4k) 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.064-066
# 데이터셋 출처: https://github.com/warnikchow/3i4k (fci_train_val.txt / fsi.txt 등)

import nltk
import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.model_selection import train_test_split


def read_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = [line.split("\t") for line in f.read().splitlines() if "\t" in line]
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
