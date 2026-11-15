# 07. 한국어 혐오 분류 데이터셋 (Korean Extremist Website Womad Hate Speech Data)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.049–054

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 한국어 혐오 분류 데이터셋 (Korean Extremist Website Womad Hate Speech Data) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 자연어 처리 / 이진 분류(Classification) / 지도학습 |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://www.kaggle.com/datasets/captainnemo9292/korean-extremist-website-womad-hate-speech-data |
| **원문 PDF 표기 링크** | https://www.kaggle.com/captainnemo9292/korean-extremist-website-womad-hate-speech-data |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle 공식 페이지 브라우저 검증 완료) |
| **교과서 연계** | 인공지능과 미래사회 p.236 (텍스트 전처리) |

한국어 문장에 혐오 표현이 포함되어 있는지 여부를 분류하는 총 2,000문장 규모의 한국어 NLP 입문 데이터셋입니다.

---

## 2. 데이터셋 특징

- 혐오 발언 여부와 한국어 문장 2개 속성, 총 **2,000개 문장**(혐오 발언 50%, 일반 문장 50%)으로 구성됩니다.
- **한국어 NLP 특성**: 영어와 달리 어순보다 조사의 역할이 크고, 띄어쓰기 오류·주어 생략·억양 의존 문장이 많아 형태소 분석(`KoNLPy Okt`) 전처리가 필수적입니다.
- **편집 참고**: p.049에 `데이터셋 목적` 라벨이 2번 중복 인쇄되어 있으며, 두 번째 항목은 실제로 `데이터셋 특징` 내용입니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 값 |
|---|---|---|---|
| ① | 혐오 발언 여부 | 혐오 발언 여부 라벨 | `0`: 일반, `1`: 혐오 발언 |
| ② | 문장 | 한국어 텍스트 문장 | 문자열 (총 2,000건) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 2,000개 문장과 라벨(0/1) 표.
- **전처리 데이터**: 형태소(`Okt`) 토큰화 후 빈도 상위 4,885개 단어 기준 Bag-of-Words(BoW) 이진/빈도 벡터 표.

---

## 5. 데이터셋 활용

1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **한국어 혐오 분류를 위한 신경망 모델 설계하기**.
3. **학습데이터 추가하기**: 엉터리 데이터 등 새 데이터를 추가해 모델 성능 변화와 데이터 품질의 중요성 탐구.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: Kaggle 데이터셋 페이지는 로그인 또는 Kaggle CLI(`kaggle datasets download -d captainnemo9292/korean-extremist-website-womad-hate-speech-data`)를 통해 `hate_speech_data.csv`를 내려받아야 합니다(단순 `urllib` 호출은 불가). 실행 시 `konlpy`, `nltk` 설치가 필요합니다.

```python
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
```
