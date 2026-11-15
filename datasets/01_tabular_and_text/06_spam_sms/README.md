# 06. 스팸 SMS 분류 데이터셋 (SMS Spam Collection)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.043–048

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 스팸 SMS 분류 데이터셋 (SMS Spam Collection) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 자연어 처리 / 이진 분류(Classification) / 지도학습 |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | CSV 파일 |
| **데이터셋 링크** | https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| **원문 PDF 표기 링크** | https://archive.ics.uci.edu/ml/datasets/sms+spam+collection |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, UCI 신규 경로 리다이렉트) |
| **교과서 연계** | 인공지능과 미래사회 p.236 (텍스트 전처리), p.240 (순환 신경망 분류) |

영문 SMS 문자가 정상(ham)인지 스팸(spam)인지 판별하는 이진 텍스트 분류 입문용 데이터셋입니다.

---

## 2. 데이터셋 특징

- SMS 메시지와 스팸 여부 2개 속성, 총 **5,572개** 메시지(정상 87%, 스팸 13%)로 구성되며, 중복 제거 시 5,169개입니다.
- **텍스트 전처리 핵심**: 토큰화(Tokenization), 정규화(Normalization), 정수 인코딩 및 패딩(`pad_sequences`, `maxlen=100`) 과정을 학습합니다.

---

## 3. 데이터셋 구성

| 번호 | 속성명 | 설명 | 범위 / 예시 |
|---|---|---|---|
| ① | 스팸 여부 (`v1`) | 정상/스팸 라벨 | `ham`: 일반 SMS, `spam`: 스팸 SMS |
| ② | SMS (`v2`) | 영문 문자 내용 | 예: `"Oh k...i'm watching here:)"` |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 5,572개 행 중 중복 제거된 5,169개 데이터.
- **전처리 데이터**: 정수 인덱스로 치환되고 `maxlen=100`으로 제로 패딩된 100차원 숫자 시퀀스 표.

---

## 5. 데이터셋 활용

1. **텍스트 데이터 전처리 과정 이해하기**: 원-핫 인코딩 및 정수 인코딩 비교 실습 (`인공지능과 미래사회 p.236`).
2. **순환 신경망(RNN)으로 텍스트 데이터 분류해보기**: 스팸 SMS 판별 모델 설계 (`인공지능과 미래사회 p.240`).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 코드 내 예시 GitHub raw URL(`mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv`)은 현재도 정상 작동(200 OK)합니다.

```python
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
```
