# 01. MNIST 데이터셋 (MNIST Datasets)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.076–080

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | MNIST 데이터셋 (MNIST Datasets) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | jpg 파일 |
| **데이터셋 링크** | https://storage.googleapis.com/cvdf-datasets/mnist/ |
| **원문 PDF 표기 링크** | http://yann.lecun.com/exdb/mnist/ (원본 파일 404로 공식 미러 대체) |
| **링크 검증 상태 (2026-11)** | 대체 완료 (200 OK, CVDF 공식 GCP 미러 / PyTorch S3 미러: https://ossci-datasets.s3.amazonaws.com/mnist/) |
| **교과서 연계** | 없음 |

0부터 9까지의 손글씨 숫자를 10개 클래스로 분류하는 28×28 픽셀 흑백 이미지 데이터셋으로, 컴퓨터 비전 입문 분야에서 가장 널리 활용됩니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: 가로·세로 **28×28 픽셀 흑백(1채널)** 이미지로 이루어져 학습이 매우 빠르고 간단합니다.
- **규모 및 원본과의 차이**: 원본은 학습용 60,000장 + 테스트용 10,000장 = 총 **70,000장**이며, 예시 코드에서는 입문자 접근성을 위해 클래스별 학습 500장·테스트 100장(총 6,000장)만 `.jpg`로 추출해 저장합니다.

---

## 3. 데이터셋 구성

| 클래스 라벨 | 대상 | 구성 |
|---|---|---|
| `0` ~ `9` (총 10개 클래스) | 손글씨 숫자 0~9 | 28×28 흑백 이미지 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `train/`과 `test/` 폴더 하위에 `0`~`9` 클래스별 폴더가 생성되고 각 폴더에 `.jpg` 이미지가 저장됩니다.

---

## 5. 데이터셋 활용

1. **온전한 학습을 통해 MNIST 데이터 분류하기**: 전체 10개 클래스 학습 후 테스트 성능 확인.
2. **부분 학습을 통해 MNIST 데이터 분류하기**: 일부 클래스만 학습시킨 뒤 미학습 클래스까지 테스트하며 데이터의 중요성과 한계점 분석.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문(p.079~080)은 `tf.keras.datasets.mnist.load_data()`를 사용해 데이터를 내려받아 클래스별 폴더(`./train/{0..9}/`, `./test/{0..9}/`)에 `jpg`로 저장합니다. 참고: 원문 제목/활용에 'MINST' 오타가 1회 등장합니다.

```python
# [I01] MNIST 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.079-080
# 참고: 원본 Yann LeCun 서버의 .gz 파일이 404이므로 tf.keras 내장 로더 또는
#       공식 CVDF 미러(https://storage.googleapis.com/cvdf-datasets/mnist/)를 활용하세요.

import os
import cv2 as cv
import numpy as np
import tensorflow as tf


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def main():
    # 1. MNIST 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스(0~9)별로 train 500장, test 100장씩 jpg 저장
    for i in range(10):
        train_path = f"./train/{i}/"
        test_path = f"./test/{i}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)]
        test_images = x_test[np.where(y_test == i)]

        for j, train_img in enumerate(train_images):
            if j < 500:
                cv.imwrite(f"./train/{i}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{i}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
```
