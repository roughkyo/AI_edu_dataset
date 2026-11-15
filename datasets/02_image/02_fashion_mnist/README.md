# 02. Fashion MNIST 데이터셋 (Fashion MNIST Datasets)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.081–086

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Fashion MNIST 데이터셋 (Fashion MNIST Datasets) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | jpg 파일 |
| **데이터셋 링크** | https://github.com/zalandoresearch/fashion-mnist |
| **원문 PDF 표기 링크** | https://github.com/zalandoresearch/fashion-mnist |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소) |
| **교과서 연계** | 없음 |

티셔츠, 바지, 니트, 드레스 등 10가지 의류·패션 품목을 28×28 흑백 이미지로 구성한 데이터셋으로, MNIST보다 변별 난이도가 약간 높은 벤치마크입니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: 28×28 픽셀 흑백 이미지, 총 **10개 의류 클래스**.
- **규모 및 원본과의 차이**: 원본은 학습 60,000장 + 테스트 10,000장 = 총 **70,000장**이며, 예시 코드에서 클래스별 학습 600장·테스트 100장으로 축소 추출합니다.
- **편집 참고**: 책 목차(p.2)에서는 `3) Fashion MNIST _ 087`로 표기되었으나, 실제 본문은 2번(p.081) 위치에 수록되어 있습니다.

---

## 3. 데이터셋 구성

| 번호 | 클래스명 | 설명 |
|---|---|---|
| 0 | `T-shirt` | 티셔츠(T-shirt/top) 28×28 흑백 이미지 |
| 1 | `Trouser` | 바지 28×28 흑백 이미지 |
| 2 | `Pullover` | 니트/풀오버 28×28 흑백 이미지 |
| 3 | `Dress` | 드레스 28×28 흑백 이미지 |
| 4 | `Coat` | 외투 28×28 흑백 이미지 |
| 5 | `Sandal` | 샌들 28×28 흑백 이미지 |
| 6 | `Shirt` | 셔츠 28×28 흑백 이미지 |
| 7 | `Sneaker` | 운동화 28×28 흑백 이미지 |
| 8 | `Bag` | 가방 28×28 흑백 이미지 |
| 9 | `Ankel boot` (Ankle boot) | 앵클부츠 28×28 흑백 이미지 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `train/`과 `test/` 폴더 아래 의류 클래스명 폴더별로 `.jpg` 이미지가 저장됩니다.

---

## 5. 데이터셋 활용

1. **온전한 학습을 통해 Fashion MNIST 데이터 분류하기**.
2. **부분 학습을 통해 Fashion MNIST 데이터 분류하기**.
3. **(심화) 생성 네트워크(GAN)를 활용하여 이미지 생성하기**: 생성자(Generator)와 판별자(Discriminator)의 적대적 학습 원리 탐구.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.085~086)는 `tf.keras.datasets.fashion_mnist.load_data()`를 사용해 클래스별 폴더에 `jpg`를 저장합니다. 원문 코드에서 `np.where(y_train == i)[0]`으로 되어 있던 1차원 인덱싱을 정상 동작하도록 정리했습니다.

```python
# [I02] Fashion MNIST 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.085-086

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
    names = [
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankel boot",
    ]

    # 1. Fashion MNIST 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스별 train 600장, test 100장 jpg 저장
    for i in range(10):
        train_path = f"./train/{names[i]}/"
        test_path = f"./test/{names[i]}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)[0]]
        test_images = x_test[np.where(y_test == i)[0]]

        for j, train_img in enumerate(train_images):
            if j < 600:
                cv.imwrite(f"./train/{names[i]}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{names[i]}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
```
