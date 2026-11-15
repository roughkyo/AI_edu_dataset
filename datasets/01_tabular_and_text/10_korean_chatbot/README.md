# 10. 한국어 챗봇 데이터셋 (Chatbot_data (Korean Chatbot Q&A))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.067–073

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 한국어 챗봇 데이터셋 (Chatbot_data (Korean Chatbot Q&A)) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | Question Answering, Text Generation / 지도학습 |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://github.com/songys/Chatbot_data |
| **원문 PDF 표기 링크** | https://github.com/songys/Chatbot_data |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소) |
| **교과서 연계** | 인공지능과 미래사회 p.236 (텍스트 전처리) |

인간의 질문(Q)과 챗봇의 대답(A) 11,876쌍 및 대화 상황 라벨(일상/이별/사랑)로 구성된 한국어 대화 생성·질의응답 데이터셋입니다.

---

## 2. 데이터셋 특징

- 질문(`Q`), 대답(`A`), 상황 라벨(`label`) 3개 열, 총 **11,876쌍**으로 구성됩니다.
- **전처리 포인트**: 질문과 대답의 전체 어휘를 합쳐 상위 2,000개 공통 토큰으로 사전을 만든 뒤, 질문 2,000차원 + 대답 2,000차원 = 총 4,000차원 벡터로 병합 저장하는 예제를 제시합니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 값 / 범위 |
|---|---|---|---|
| ① | `Q` | 인간의 질문 문장 | 한국어 텍스트 |
| ② | `A` | 챗봇의 대답 문장 | 한국어 텍스트 |
| ③ | `label` | 대화 상황 카테고리 | `0`: 일상다반사, `1`: 이별(부정), `2`: 사랑(긍정) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 11,876행 × 3열 CSV (`Q`, `A`, `label`).
- **전처리 데이터**: 질문 2,000차원 + 대답 2,000차원 = 4,000열 벡터 CSV (`K_chatbot_train.csv`, `K_chatbot_test.csv`).

---

## 5. 데이터셋 활용

1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **텍스트 인식과 대답 생성 모델에 대해 알아보기**: 단순 분류 모델이 아니라 연속적 문장 생성 모델(Seq2Seq 등)의 필요성과 구조 탐구.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.070~073)는 `Korpora` 패키지를 통해 데이터를 내려받으며, 쉘 명령(`!git clone`, `pip install Korpora`)을 주석으로 정리했습니다.

```python
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
```
