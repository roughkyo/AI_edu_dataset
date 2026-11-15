# 09. 한국어 의도 분류 데이터셋 (3i4k (Intention Identification in Korean))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.061–066

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 한국어 의도 분류 데이터셋 (3i4k (Intention Identification in Korean)) |
| **카테고리** | 수치/텍스트 |
| **데이터셋 목적** | 자연어 처리 / 다중 분류(7개 발화 의도) / 지도학습 |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | TXT 파일 |
| **데이터셋 링크** | https://github.com/warnikchow/3i4k |
| **원문 PDF 표기 링크** | https://github.com/warnikchow/3i4k |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소) |
| **교과서 연계** | 인공지능과 미래사회 p.236 (텍스트 전처리) |

한국어 발화가 가진 7가지 의도(불완전 발화, 서술, 질문, 요구, 수사 의문, 수사 명령, 억양 의존)를 분류하는 55,134문장 규모의 다중 분류 데이터셋입니다.

---

## 2. 데이터셋 특징

- 한국어 문장과 0~6까지의 의도 라벨 2개 속성, 총 **55,134개 문장**으로 구성됩니다.
- **편집 참고**: p.061 도입문과 p.063 예시 설명에 "전체 2,000개의 문장"(직전 T07 한국어 혐오 분류 데이터셋 문구 잔재)이라는 오기가 있으나, 실제 데이터셋 특징 및 원 논문/GitHub 기준은 55,134개(50,000건 이상) 문장입니다.

---

## 3. 데이터셋 구성

| 라벨 | 의도 분류 | 설명 |
|---|---|---|
| `0` | Fragments | 불완전하게 끝난 문장 / 조각 발화 |
| `1` | 서술 (Statement) | 다른 분류에 속하지 않는 일반 평서문 |
| `2` | 질문 (Question) | 상대에게 답변을 요구하는 의문문 |
| `3` | 요구 (Command) | 상대에게 행위를 요구하는 명령/청유문 |
| `4` | 수사 의문 (Rhetorical Question) | 의문문 형태이나 답변을 요구하지 않는 문장 |
| `5` | 수사 명령 (Rhetorical Command) | 명령형 형태이나 행위를 강요하지 않는 문장 (예: "즐거운 하루 보내세요") |
| `6` | 억양 의존 (Intonation-dependent) | 억양에 따라 의도가 달라지는 발화 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `라벨\t문장` 형태의 탭 구분 TXT 파일 (`fsi.txt`).
- **전처리 데이터**: `Okt` 토큰화 후 상위 5,000개 토큰 기준 5,000차원 BoW + 라벨(총 5,001열) 벡터 표.

---

## 5. 데이터셋 활용

1. **텍스트 데이터 전처리 과정 이해하기** (`인공지능과 미래사회 p.236`).
2. **한국어 의도 분류를 위한 신경망 모델 설계하기** (※ 본문 중 '혐오 분류' 오기 존재).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.064~066)에서 `line.split('\t')`의 탭 문자 표기가 좁아져 있으며, `kor_train.csv`/`kor_test.csv` 저장 과정을 Python 표준 문법에 맞게 정확히 복원했습니다.

```python
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
```
