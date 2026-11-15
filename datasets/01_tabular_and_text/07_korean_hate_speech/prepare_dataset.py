# [T07] 한국어 혐오 분류 데이터셋 전처리 예제 (BoW 벡터화)
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
