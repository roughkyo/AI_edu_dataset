# 03. CIFAR10 데이터셋 (CIFAR-10 Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.087–091

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | CIFAR10 데이터셋 (CIFAR-10 Dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | jpg 파일 |
| **데이터셋 링크** | https://www.cs.toronto.edu/~kriz/cifar.html |
| **원문 PDF 표기 링크** | https://www.cs.toronto.edu/~kriz/cifar.html |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, University of Toronto 공식 페이지) |
| **교과서 연계** | 없음 |

비행기, 자동차, 새, 고양이 등 10개 사물·동물 클래스를 32×32 컬러 이미지로 담은 대표적 자연 이미지 분류 데이터셋입니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: 가로·세로 **32×32 픽셀 3채널 컬러(RGB)** 이미지.
- **규모 및 원본과의 차이**: 원본은 학습 50,000장 + 테스트 10,000장 = 총 **60,000장**이며, 예시 코드에서는 클래스별 학습 500장·테스트 100장만 추출합니다.
- **편집 참고**: 책 목차(p.2)에서는 `2) CIFAR10 _ 081`로 표기되었으나 실제 본문은 3번(p.087) 위치에 수록되어 있으며, p.089 `[그림 2]` 캡션에 'Fashion MNIST' 오기가 있습니다.

---

## 3. 데이터셋 구성

| 번호 | 클래스명 | 설명 |
|---|---|---|
| 0 | `Airplane` (비행기) | 32×32 컬러 이미지 |
| 1 | `Automobile` (자동차) | 32×32 컬러 이미지 |
| 2 | `Bird` (새) | 32×32 컬러 이미지 |
| 3 | `Cat` (고양이) | 32×32 컬러 이미지 |
| 4 | `Deer` (사슴) | 32×32 컬러 이미지 |
| 5 | `Dog` (개) | 32×32 컬러 이미지 |
| 6 | `Frog` (개구리) | 32×32 컬러 이미지 |
| 7 | `Horse` (말) | 32×32 컬러 이미지 |
| 8 | `Ship` (배) | 32×32 컬러 이미지 |
| 9 | `Truck` (트럭) | 32×32 컬러 이미지 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `train/`, `test/` 아래 10개 영문 클래스명 폴더별 `.jpg` 이미지 저장.

---

## 5. 데이터셋 활용

1. **전체 학습을 통한 CIFAR10 이미지 분류하기**.
2. **부분 학습을 통한 CIFAR10 이미지 분류하기**.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.090~091)는 `tf.keras.datasets.cifar10.load_data()`에서 `(x_train, y_train)`을 불러와 `np.where(y_train == i)[0]`으로 클래스별 `jpg` 파일을 저장합니다(OpenCV 저장 시 RGB→BGR 변환 참고 주석 추가).

```python
# [I03] CIFAR10 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.090-091

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
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck",
    ]

    # 1. CIFAR10 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스별 train 500장, test 100장 jpg 저장 (y_train은 (N, 1) 형태이므로 [0] 인덱싱)
    for i in range(10):
        train_path = f"./train/{names[i]}/"
        test_path = f"./test/{names[i]}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)[0]]
        test_images = x_test[np.where(y_test == i)[0]]

        for j, train_img in enumerate(train_images):
            if j < 500:
                # 참고: OpenCV는 BGR 순서를 사용하므로 색상을 보존하려면 cv.cvtColor(img, cv.COLOR_RGB2BGR) 권장
                cv.imwrite(f"./train/{names[i]}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{names[i]}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
```
