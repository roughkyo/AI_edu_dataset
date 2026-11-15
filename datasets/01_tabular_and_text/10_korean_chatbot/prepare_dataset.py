# [T10] 한국어 챗봇 데이터셋 (Chatbot_data) 전처리 예제
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
