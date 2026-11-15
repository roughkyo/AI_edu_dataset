# 08. 네이버 영화 리뷰데이터(NSMC) (Naver Sentiment Movie Corpus (NSMC))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.055–060

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 네이버 영화 리뷰데이터(NSMC) (Naver Sentiment Movie Corpus (NSMC)) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 자연어 처리 / 이진 분류(감성 분석) / 지도학습 |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | TXT 파일 |
| **데이터셋 링크** | https://github.com/e9t/nsmc |
| **원문 PDF 표기 링크** | https://github.com/e9t/nsmc.git |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소) |
| **교과서 연계** | 없음 |

네이버 영화 리뷰 텍스트를 긍정(1) 또는 부정(0)으로 분류한 총 20만 건의 대규모 한국어 감성 분석 표준 벤치마크 데이터셋입니다.

---

## 2. 데이터셋 특징

- 리뷰 고유 key(`id`), 리뷰 텍스트(`document`), 감성 라벨(`label`) 3개 속성으로 구성됩니다.
- **데이터 규모**: `train` 150,000개, `test` 50,000개 등 총 **200,000개** 리뷰를 포함합니다(전체 등장 토큰 약 534,534개, 예제 코드에서는 빈도 상위 5,000개 선택).

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 예시 / 값 |
|---|---|---|---|
| ① | `id` | 리뷰 고유 식별 정수 | 예: `9976970` |
| ② | `document` | 한국어 영화 리뷰 문장 | 예: `"청춘 영화의 최고봉..."` |
| ③ | `label` | 긍정/부정 감성 라벨 | `0`: 부정, `1`: 긍정 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 탭(`	`)으로 구분된 150,000행 × 3열(`ratings_train.txt`) 및 50,000행 × 3열(`ratings_test.txt`).

---

## 5. 데이터셋 활용

1. **한글 텍스트 데이터셋 전처리(정규화)하기**:
   - Q1-1 / Q1-2: 글자 단위 vs 단어(형태소) 단위 토큰화 성능 및 실행 시간 비교.
2. **텍스트의 감정 분류하기**:
   - Q2-1: `vocab_size` 변화에 따른 성능 변화 분석.
   - Q2-2: 이진 분류 임곗값(threshold) 변화 실험.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.057~058) 상단에 BERT 관련 import(`transformers`, `torch`, `keras`, `sklearn`)와 하단에 `Okt` + BoW 전처리 코드가 함께 수록되어 있습니다. 쉘 명령(`!pip install`, `!git clone`)은 주석으로 보존했습니다.

```python
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
```
