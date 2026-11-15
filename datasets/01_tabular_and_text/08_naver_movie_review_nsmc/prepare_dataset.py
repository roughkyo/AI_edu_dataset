# [T08] 네이버 영화 리뷰 데이터셋 (NSMC) 전처리 예제
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
    train = pd.read_csv("nsmc/ratings_train.txt", sep="\t")
    test = pd.read_csv("nsmc/ratings_test.txt", sep="\t")

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
