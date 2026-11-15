from pathlib import Path

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
BASE = ROOT / "datasets" / "02_image"
BASE.mkdir(parents=True, exist_ok=True)

ITEMS = [
    {
        "slug": "01_mnist",
        "num": "01",
        "title_kr": "MNIST 데이터셋",
        "title_en": "MNIST Datasets",
        "pages": "p.076–080",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "jpg 파일",
        "category": "이미지",
        "purpose": "이미지 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://storage.googleapis.com/cvdf-datasets/mnist/",
        "url_pdf": "http://yann.lecun.com/exdb/mnist/ (원본 파일 404로 공식 미러 대체)",
        "url_status": "대체 완료 (200 OK, CVDF 공식 GCP 미러 / PyTorch S3 미러: https://ossci-datasets.s3.amazonaws.com/mnist/)",
        "textbook": "없음",
        "summary": "0부터 9까지의 손글씨 숫자를 10개 클래스로 분류하는 28×28 픽셀 흑백 이미지 데이터셋으로, 컴퓨터 비전 입문 분야에서 가장 널리 활용됩니다.",
        "features": """- **해상도 및 채널**: 가로·세로 **28×28 픽셀 흑백(1채널)** 이미지로 이루어져 학습이 매우 빠르고 간단합니다.
- **규모 및 원본과의 차이**: 원본은 학습용 60,000장 + 테스트용 10,000장 = 총 **70,000장**이며, 예시 코드에서는 입문자 접근성을 위해 클래스별 학습 500장·테스트 100장(총 6,000장)만 `.jpg`로 추출해 저장합니다.""",
        "structure": """| 클래스 라벨 | 대상 | 구성 |
|---|---|---|
| `0` ~ `9` (총 10개 클래스) | 손글씨 숫자 0~9 | 28×28 흑백 이미지 |""",
        "example": """- **원 데이터(Raw Data)**: `train/`과 `test/` 폴더 하위에 `0`~`9` 클래스별 폴더가 생성되고 각 폴더에 `.jpg` 이미지가 저장됩니다.""",
        "utilization": """1. **온전한 학습을 통해 MNIST 데이터 분류하기**: 전체 10개 클래스 학습 후 테스트 성능 확인.
2. **부분 학습을 통해 MNIST 데이터 분류하기**: 일부 클래스만 학습시킨 뒤 미학습 클래스까지 테스트하며 데이터의 중요성과 한계점 분석.""",
        "code_note": "원문(p.079~080)은 `tf.keras.datasets.mnist.load_data()`를 사용해 데이터를 내려받아 클래스별 폴더(`./train/{0..9}/`, `./test/{0..9}/`)에 `jpg`로 저장합니다. 참고: 원문 제목/활용에 'MINST' 오타가 1회 등장합니다.",
        "code": """# [I01] MNIST 데이터셋 추출 및 폴더 저장 예제
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
""",
    },
    {
        "slug": "02_fashion_mnist",
        "num": "02",
        "title_kr": "Fashion MNIST 데이터셋",
        "title_en": "Fashion MNIST Datasets",
        "pages": "p.081–086",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "jpg 파일",
        "category": "이미지",
        "purpose": "이미지 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://github.com/zalandoresearch/fashion-mnist",
        "url_pdf": "https://github.com/zalandoresearch/fashion-mnist",
        "url_status": "유효 (200 OK, GitHub 공식 저장소)",
        "textbook": "없음",
        "summary": "티셔츠, 바지, 니트, 드레스 등 10가지 의류·패션 품목을 28×28 흑백 이미지로 구성한 데이터셋으로, MNIST보다 변별 난이도가 약간 높은 벤치마크입니다.",
        "features": """- **해상도 및 채널**: 28×28 픽셀 흑백 이미지, 총 **10개 의류 클래스**.
- **규모 및 원본과의 차이**: 원본은 학습 60,000장 + 테스트 10,000장 = 총 **70,000장**이며, 예시 코드에서 클래스별 학습 600장·테스트 100장으로 축소 추출합니다.
- **편집 참고**: 책 목차(p.2)에서는 `3) Fashion MNIST _ 087`로 표기되었으나, 실제 본문은 2번(p.081) 위치에 수록되어 있습니다.""",
        "structure": """| 번호 | 클래스명 | 설명 |
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
| 9 | `Ankel boot` (Ankle boot) | 앵클부츠 28×28 흑백 이미지 |""",
        "example": """- **원 데이터(Raw Data)**: `train/`과 `test/` 폴더 아래 의류 클래스명 폴더별로 `.jpg` 이미지가 저장됩니다.""",
        "utilization": """1. **온전한 학습을 통해 Fashion MNIST 데이터 분류하기**.
2. **부분 학습을 통해 Fashion MNIST 데이터 분류하기**.
3. **(심화) 생성 네트워크(GAN)를 활용하여 이미지 생성하기**: 생성자(Generator)와 판별자(Discriminator)의 적대적 학습 원리 탐구.""",
        "code_note": "원문 코드(p.085~086)는 `tf.keras.datasets.fashion_mnist.load_data()`를 사용해 클래스별 폴더에 `jpg`를 저장합니다. 원문 코드에서 `np.where(y_train == i)[0]`으로 되어 있던 1차원 인덱싱을 정상 동작하도록 정리했습니다.",
        "code": """# [I02] Fashion MNIST 데이터셋 추출 및 폴더 저장 예제
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
""",
    },
    {
        "slug": "03_cifar10",
        "num": "03",
        "title_kr": "CIFAR10 데이터셋",
        "title_en": "CIFAR-10 Dataset",
        "pages": "p.087–091",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "jpg 파일",
        "category": "이미지",
        "purpose": "이미지 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "url_pdf": "https://www.cs.toronto.edu/~kriz/cifar.html",
        "url_status": "유효 (200 OK, University of Toronto 공식 페이지)",
        "textbook": "없음",
        "summary": "비행기, 자동차, 새, 고양이 등 10개 사물·동물 클래스를 32×32 컬러 이미지로 담은 대표적 자연 이미지 분류 데이터셋입니다.",
        "features": """- **해상도 및 채널**: 가로·세로 **32×32 픽셀 3채널 컬러(RGB)** 이미지.
- **규모 및 원본과의 차이**: 원본은 학습 50,000장 + 테스트 10,000장 = 총 **60,000장**이며, 예시 코드에서는 클래스별 학습 500장·테스트 100장만 추출합니다.
- **편집 참고**: 책 목차(p.2)에서는 `2) CIFAR10 _ 081`로 표기되었으나 실제 본문은 3번(p.087) 위치에 수록되어 있으며, p.089 `[그림 2]` 캡션에 'Fashion MNIST' 오기가 있습니다.""",
        "structure": """| 번호 | 클래스명 | 설명 |
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
| 9 | `Truck` (트럭) | 32×32 컬러 이미지 |""",
        "example": """- **원 데이터(Raw Data)**: `train/`, `test/` 아래 10개 영문 클래스명 폴더별 `.jpg` 이미지 저장.""",
        "utilization": """1. **전체 학습을 통한 CIFAR10 이미지 분류하기**.
2. **부분 학습을 통한 CIFAR10 이미지 분류하기**.""",
        "code_note": "원문 코드(p.090~091)는 `tf.keras.datasets.cifar10.load_data()`에서 `(x_train, y_train)`을 불러와 `np.where(y_train == i)[0]`으로 클래스별 `jpg` 파일을 저장합니다(OpenCV 저장 시 RGB→BGR 변환 참고 주석 추가).",
        "code": """# [I03] CIFAR10 데이터셋 추출 및 폴더 저장 예제
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
""",
    },
    {
        "slug": "04_dogs_vs_cats",
        "num": "04",
        "title_kr": "Dogs vs. Cats 데이터셋",
        "title_en": "Dogs vs. Cats Datasets",
        "pages": "p.092–098",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "jpg 파일",
        "category": "이미지",
        "purpose": "이미지 분류(이진 분류) / 지도학습(Supervised Learning)",
        "url": "https://www.kaggle.com/c/dogs-vs-cats/overview",
        "url_pdf": "https://www.kaggle.com/c/dogs-vs-cats/overview",
        "url_status": "유효 (200 OK, Kaggle Competition 공식 페이지)",
        "textbook": "없음",
        "summary": "가변 해상도의 RGB 컬러 이미지에서 개(dog)와 고양이(cat)를 구분하는 대표적 이진 이미지 분류 데이터셋입니다.",
        "features": """- **해상도 및 채널**: 이미지마다 가로·세로 크기가 다르며 RGB 3채널 컬러(`0~255`)로 구성됩니다.
- **규모**: 학습용 `train.zip` 25,000장 + 테스트용 `test1.zip` 12,500장 = 총 **37,500장**.
- **전처리 핵심**: `test1.zip`에는 정답 파일명이 없으므로 `train.zip`(개 12,500장, 고양이 12,500장)을 학습 22,000장 / 테스트(검증) 3,000장으로 분할하고 `torchvision.transforms`로 크기를 통일합니다.""",
        "structure": """| 라벨 | 클래스 | 설명 |
|---|---|---|
| `0` | `dog` | 개 이미지 (학습 12,500장 중 11,000장 train / 1,500장 test 분할) |
| `1` | `cat` | 고양이 이미지 (학습 12,500장 중 11,000장 train / 1,500장 test 분할) |""",
        "example": """- **원 데이터(Raw Data)**: 다양한 비율의 개·고양이 실사 JPG 이미지(`dog.0.jpg`~`dog.12499.jpg`, `cat.0.jpg`~`cat.12499.jpg`).""",
        "utilization": """1. **이미지 시각화하기**: 전처리된 `DataLoader` 배치를 4×6 바둑판 그리드로 시각화.
2. **학습/테스트 전처리**: `Resize(256, 256)`, `RandomCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`.
3. **학습·테스트 데이터 분류(7:3, 8:2)** 및 **분류 네트워크 설계하기**.""",
        "code_note": "원문 코드(p.095)의 Google Drive URL(`1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD`)은 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.zip`을 내려받아 `./train/`에 압축 해제한 뒤 실행하세요.",
        "code": """# [I04] Dogs vs. Cats 데이터셋 PyTorch CustomDataset 및 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.094-098
# 주의: 원문 Google Drive 링크는 만료(404)되었으므로 Kaggle에서 train.zip을 내려받아 압축 해제 후 사용하세요.

import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, files, root, mode="train", transform=None):
        self.files = files
        self.root = root
        self.mode = mode
        self.transform = transform
        if "dog" in files[0]:
            self.label = 0
        else:
            self.label = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, self.files[index]))
        if self.transform:
            img = self.transform(img)
        if self.mode == "train":
            return img, np.array([self.label])
        else:
            return img, self.files[index]


def main():
    # 1. 분할된 파일이 저장될 경로 준비 (train.zip을 ./train 에 압축 해제한 상태 기준)
    train_dir = os.path.join(os.getcwd(), "train")
    test_dir = os.path.join(os.getcwd(), "test")

    train_set_dir = os.path.join(train_dir, "train")
    os.makedirs(train_set_dir, exist_ok=True)
    test_set_dir = os.path.join(test_dir, "test")
    os.makedirs(test_set_dir, exist_ok=True)

    dog_files = [f"dog.{i}.jpg" for i in range(12500)]
    cat_files = [f"cat.{i}.jpg" for i in range(12500)]

    # 2. 개/고양이 이미지를 train(각 11,000장) / valid(각 1,500장) 폴더로 이동
    for dog, cat in zip(dog_files[:11000], cat_files[:11000]):
        shutil.move(os.path.join(train_dir, dog), os.path.join(train_set_dir, dog))
        shutil.move(os.path.join(train_dir, cat), os.path.join(train_set_dir, cat))

    for dog, cat in zip(dog_files[11000:12500], cat_files[11000:12500]):
        shutil.move(os.path.join(train_dir, dog), os.path.join(test_set_dir, dog))
        shutil.move(os.path.join(train_dir, cat), os.path.join(test_set_dir, cat))

    # 3. 이미지 전처리 파이프라인 정의
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((224, 244)),
        transforms.ToTensor(),
    ])

    # 4. Dataset 및 DataLoader 생성
    train_dog_dataset = CustomDataset(dog_files[:11000], train_set_dir, transform=train_transform)
    train_cat_dataset = CustomDataset(cat_files[:11000], train_set_dir, transform=train_transform)
    test_dog_dataset = CustomDataset(dog_files[11000:], test_set_dir, transform=test_transform)
    test_cat_dataset = CustomDataset(cat_files[11000:], test_set_dir, transform=test_transform)

    train_dataset = ConcatDataset([train_dog_dataset, train_cat_dataset])
    test_dataset = ConcatDataset([test_dog_dataset, test_cat_dataset])

    print(f"number of train dataset : {len(train_dataset)}")
    print(f"number of test dataset : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # 5. 전처리 완료된 이미지 24장 시각화
    samples, labels = next(iter(train_loader))
    classes = {0: "dog", 1: "cat"}
    fig = plt.figure(figsize=(16, 24))
    for i in range(24):
        a = fig.add_subplot(4, 6, i + 1)
        a.set_title(classes[labels[i].item()])
        a.axis("off")
        a.imshow(np.transpose(samples[i].numpy(), (1, 2, 0)))
    plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)
    plt.show()


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "05_svhn",
        "num": "05",
        "title_kr": "SVHN 데이터셋",
        "title_en": "Street View House Numbers (SVHN)",
        "pages": "p.099–105",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "png+mat 파일",
        "category": "이미지",
        "purpose": "이미지 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "http://ufldl.stanford.edu/housenumbers/",
        "url_pdf": "http://ufldl.stanford.edu/housenumbers",
        "url_status": "유효 (200 OK, Stanford UFLDL 공식 페이지)",
        "textbook": "없음",
        "summary": "Google Street View에서 수집한 실제 건물 번호판·표지판 숫자(0~9)를 담은 32×32 RGB 컬러 이미지 데이터셋으로, MNIST의 실사 심화 버전입니다.",
        "features": """- **해상도 및 채널**: Format 2 기준 **32×32 픽셀 RGB 3채널** 이미지(한 사진 안에 인접 숫자가 일부 포함되기도 함).
- **규모**: 학습용 73,257개, 테스트용 26,032개, 추가 난이도 낮은 데이터(`extra`) 531,131개.
- **원본 포맷 2종**: Format 1(원본 해상도 PNG + `digitStruct.mat` 바운딩 박스), Format 2(32×32 크롭된 `train_32x32.mat`, `test_32x32.mat` — `torchvision` 기본 지원).""",
        "structure": """| 라벨 | 대상 | 비고 |
|---|---|---|
| `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` (10클래스) | 실제 거리 번호판 숫자 | Format 2 기준 32×32 RGB |""",
        "example": """- **Format 1**: 건물 전체 표지판 이미지와 숫자 영역(Bounding Box).
- **Format 2**: 32×32로 균일하게 잘라낸 숫자 타일 이미지(`X`, `y` 행렬).""",
        "utilization": """1. **학습데이터, 테스트 데이터를 위한 DataLoader 만들기**: `torchvision.transforms`의 다양한 데이터 증강 실험(Q1-1).
2. **분류 모델 정량적 성능 파악 (Quantitative)**: CNN 설계, 학습, 모델 저장 및 테스트 정확도 산출(Q2-1~Q3-2).
3. **분류 모델 정성적 성능 파악 (Qualitative)**: 예측 결과와 정답 라벨·이미지 직접 비교 시각화(Q4-1~Q4-3).""",
        "code_note": "원문 코드(p.102~104)는 `torchvision.datasets.SVHN`으로 Format 2(`.mat`)를 자동 다운로드하고 `DataLoader` 학습/평가 루프 템플릿을 제공합니다(오타 `transfrom`→`transform` 수정).",
        "code": """# [I05] SVHN 데이터셋 PyTorch 로드 및 학습/평가 루프 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.102-104

import torch
from torchvision import datasets, transforms


def main():
    # 1. 저장 경로 및 전처리 정의
    download_root = "./SVHN_DATASET"
    svhn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # 2. SVHN 데이터셋 다운로드 (split='train' / 'test')
    train_dataset = datasets.SVHN(
        root=download_root, split="train", transform=svhn_transform, download=True
    )
    test_dataset = datasets.SVHN(
        root=download_root, split="test", transform=svhn_transform, download=True
    )

    # 3. DataLoader 구성
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # 4. 학습 / 테스트 반복문 템플릿 (model 정의 후 사용)
    # model.train()
    # for epoch in range(30):
    #     for iter_idx, (image, label) in enumerate(train_dataloader):
    #         pass
    #
    # model.eval()
    # with torch.no_grad():
    #     for image, label in test_dataloader:
    #         pass


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "06_stl10",
        "num": "06",
        "title_kr": "STL-10 데이터셋",
        "title_en": "STL-10 Dataset",
        "pages": "p.106–112",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "mat/bin 파일 (추가로 png 등 저장 가능)",
        "category": "이미지",
        "purpose": "이미지 분류(지도학습) 및 비지도 표현 학습(Unsupervised Learning)",
        "url": "https://cs.stanford.edu/~acoates/stl10/",
        "url_pdf": "https://cs.stanford.edu/~acoates/stl10/",
        "url_status": "유효 (200 OK, Stanford 공식 페이지)",
        "textbook": "없음",
        "summary": "ImageNet에서 추출한 96×96 고해상도 컬러 이미지 데이터셋으로, 소량의 라벨 데이터(지도학습)와 대규모 비라벨 이미지 10만 장(비지도학습)을 함께 제공합니다.",
        "features": """- **해상도 및 채널**: **96×96 픽셀 3채널 RGB** (CIFAR-10의 32×32보다 훨씬 높은 해상도).
- **규모**:
  - **지도학습용(10클래스)**: 클래스별 학습 500장(총 5,000장) + 테스트 800장(총 8,000장).
  - **비지도학습용(`unlabeled`)**: 라벨 없는 **100,000장** (10클래스 외 곰, 토끼, 기차, 버스 등 다양한 객체 포함).""",
        "structure": """| 구분 | 클래스 / 수량 | 설명 |
|---|---|---|
| 지도학습 10클래스 | ① airplane ② bird ③ car ④ cat ⑤ deer ⑥ dog ⑦ horse ⑧ monkey ⑨ ship ⑩ truck | train 5,000장 / test 8,000장 (96×96 RGB) |
| 비지도학습 (`unlabeled`) | 라벨 없음 (다양한 동물·탈것 포함) | 100,000장 (96×96 RGB) |""",
        "example": """- **1. 지도학습용 이미지**: 10개 클래스별 실사 이미지.
- **2. 비지도학습용 이미지**: 코끼리, 사슴, 선박, 기차, 이구아나 등 폭넓은 분포의 라벨 없는 이미지.""",
        "utilization": """1. **데이터셋 분리**: `StratifiedShuffleSplit`을 이용해 `test_ds0`(8,000장)을 계층적 샘플링으로 검증(validation)과 테스트(test)로 8:2 분할(Q1-1).
2. **분류 모델 학습 및 정량 평가(Quantitative)** (Q2-1~Q3-3).
3. **(심화) 비지도 학습 개념 이해하기**: 군집화(Clustering) 및 `unlabeled` 데이터셋 로드 비교(Q4-1~Q4-2).""",
        "code_note": "원문 코드(p.109~112)는 `torchvision.datasets.STL10` 다운로드, `StratifiedShuffleSplit` 검증 분할, 그리고 `torchvision.utils.save_image`를 이용한 라벨별 PNG 추출 코드를 제공합니다.",
        "code": """# [I06] STL-10 데이터셋 로드, 계층적 분할 및 라벨별 PNG 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.109-112

import os
import torch
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torchvision import datasets, transforms


def main():
    # 1. 저장 경로 및 STL-10 다운로드
    download_root = "./DATA"
    data_transform = transforms.Compose([transforms.ToTensor()])

    train_ds = datasets.STL10(
        download_root, split="train", transform=data_transform, download=True
    )
    test_ds0 = datasets.STL10(
        download_root, split="test", transform=data_transform, download=True
    )

    # 2. test_ds0를 StratifiedShuffleSplit으로 val(80%) / test(20%) 분할
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    indices = list(range(len(test_ds0)))
    label_test0 = [label for _, label in test_ds0]

    for test_idx, val_idx in sss.split(indices, label_test0):
        val_ds = Subset(test_ds0, val_idx)
        test_ds = Subset(test_ds0, test_idx)

    # 3. DataLoader 구성
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False)

    # 4. 라벨별 폴더에 PNG 이미지로 저장하기
    for cls_idx in range(10):
        os.makedirs(f"./stl10_train_png/{cls_idx}", exist_ok=True)

    image_num = 0
    for images, labels in train_dataloader:
        for num in range(images.size()[0]):
            lbl = labels[num].item()
            torchvision.utils.save_image(images[num, :, :, :], f"./stl10_train_png/{lbl}/{image_num}.png")
            image_num += 1


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "07_pascal_voc",
        "num": "07",
        "title_kr": "PASCAL VOC 데이터셋",
        "title_en": "Pascal VOC Dataset",
        "pages": "p.113–120",
        "difficulty": "★★★★★ (5/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "jpg 파일 (+ xml 어노테이션)",
        "category": "이미지",
        "purpose": "객체 검출(Object Detection) / 지도학습",
        "url": "https://pjreddie.com/projects/pascal-voc-dataset-mirror/",
        "url_pdf": "https://pjreddie.com/projects/pascal-voc-dataset-mirror/",
        "url_status": "유효 (200 OK, pjreddie 공식 미러 및 tar 파일 정상 확인)",
        "textbook": "없음",
        "summary": "하나의 이미지 안에 여러 클래스의 객체가 동시에 등장하며, 바운딩 박스(Bounding Box)와 세그멘테이션 마스크를 제공하는 대표적 객체 검출 벤치마크입니다.",
        "features": """- **20개 객체 클래스**와 각 객체의 위치 좌표(`xmin, ymin, xmax, ymax` XML 어노테이션)를 포함합니다.
- **규모**:
  - **VOC2007**: 객체 12,608개 포함 이미지 5,011장 (예제 코드 기준 `train` 2,501장 + `val` 2,510장 / `test` 4,952장).
  - **VOC2012**: 객체 27,450개 포함 이미지 11,540장.
- **전처리 핵심**: 이미지마다 해상도와 비율이 다르므로 `albumentations`의 `LongestMaxSize` + `PadIfNeeded`를 활용해 **바운딩 박스 좌표까지 함께 보정**하며 600×600으로 통일합니다.""",
        "structure": """| 번호 범위 | 클래스 목록 (총 20개) |
|---|---|
| 0 ~ 3 | `aeroplane`, `bicycle`, `bird`, `boat` |
| 4 ~ 7 | `bottle`, `bus`, `car`, `cat` |
| 8 ~ 11 | `chair`, `cow`, `diningtable`, `dog` |
| 12 ~ 15 | `horse`, `motorbike`, `person`, `pottedplant` |
| 16 ~ 19 | `sheep`, `sofa`, `train`, `tvmonitor` |""",
        "example": """- **원 데이터(Raw Data)**: 다양한 비율의 실사 JPG 이미지 + XML 어노테이션 + 바운딩 박스 시각화.""",
        "utilization": """1. **이미지 시각화하기**: 원본 이미지 로드 확인.
2. **바운딩 박스 시각화하기**: XML에서 파싱한 다중 바운딩 박스와 클래스 라벨을 이미지 위에 그려 확인.
3. **객체 검출(Object Detection)**: 위치 추정(Localization)과 분류(Classification)를 동시에 수행하는 모델 학습 준비.""",
        "code_note": "원문 코드(p.115~120)는 Google Colab 환경 기준으로 작성되어 있으며(`!pip install -U albumentations`, `%matplotlib inline`), `VOCDetection`을 상속한 `myVOCDetection` 클래스로 XML 파싱, `albumentations` 리사이징, 바운딩 박스 시각화 및 PNG 저장을 수행합니다.",
        "code": """# [I07] PASCAL VOC 2007 데이터셋 로드, 바운딩 박스 파싱 및 Albumentations 전처리 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.115-120
# [Colab/Shell] !pip install -U albumentations

import collections
import os
import xml.etree.ElementTree as ET
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2 as ToTensor
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.datasets import VOCDetection
from torchvision.transforms.functional import to_pil_image

classes = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class myVOCDetection(VOCDetection):
    def __getitem__(self, index):
        img = np.array(Image.open(self.images[index]).convert("RGB"))
        target = self.parse_voc_xml(ET.parse(self.annotations[index]).getroot())

        targets = []
        labels = []
        for t in target["annotation"]["object"]:
            label = np.zeros(5)
            label[:] = [
                int(t["bndbox"]["xmin"]),
                int(t["bndbox"]["ymin"]),
                int(t["bndbox"]["xmax"]),
                int(t["bndbox"]["ymax"]),
                classes.index(t["name"]),
            ]
            targets.append(list(label[:4]))
            labels.append(label[4])

        if self.transforms:
            augmentations = self.transforms(image=img, bboxes=targets, category_ids=labels)
            img = augmentations["image"]
            targets = augmentations["bboxes"]

        return img, targets, labels

    def parse_voc_xml(self, node: ET.Element) -> Dict[str, Any]:
        voc_dict: Dict[str, Any] = {}
        children = list(node)
        if children:
            def_dic: Dict[str, Any] = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == "annotation":
                def_dic["object"] = [def_dic["object"]]
            voc_dict = {
                node.tag: {
                    ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()
                }
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict


def show(img, targets, labels, class_names=classes):
    img = to_pil_image(img)
    draw = ImageDraw.Draw(img)
    targets = np.array(targets)
    colors = np.random.randint(0, 255, size=(80, 3), dtype="uint8")

    for tg, label in zip(targets, labels):
        id_ = int(label)
        bbox = tg[:4]
        color = [int(c) for c in colors[id_]]
        name = class_names[id_]
        draw.rectangle(((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline=tuple(color), width=3)
        draw.text((bbox[0], bbox[1]), name, fill=(255, 255, 255, 0))

    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))
    plt.show()


def main():
    path2data = "./voc"
    os.makedirs(path2data, exist_ok=True)

    IMAGE_SIZE = 600
    train_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=IMAGE_SIZE),
            A.PadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT),
            ToTensor(),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.4, label_fields=["category_ids"]),
    )

    train_ds = myVOCDetection(path2data, year="2007", image_set="train", download=True, transforms=train_transforms)
    val_ds = myVOCDetection(path2data, year="2007", image_set="test", download=True, transforms=train_transforms)
    print(f"VOC2007 train: {len(train_ds)}, test: {len(val_ds)}")


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "08_horse_to_zebra",
        "num": "08",
        "title_kr": "Horse to Zebra 데이터셋",
        "title_en": "Horse2zebra dataset",
        "pages": "p.121–125",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "jpg+csv 파일",
        "category": "이미지",
        "purpose": "스타일 이전(Style Transfer) / 비지도학습(Unsupervised Learning)",
        "url": "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/",
        "url_pdf": "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ (404 만료로 공식 대체)",
        "url_status": "대체 완료 (200 OK, UC Berkeley CycleGAN 공식 서버 / 직접 ZIP: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip)",
        "textbook": "인공지능과 미래사회 p.241 (스타일 트랜스퍼)",
        "summary": "짝지어지지 않은(unpaired) 말(Horse)과 얼룩말(Zebra) 이미지를 활용해 CycleGAN 등의 이미지 스타일 이전 모델을 학습시키는 비지도학습 데이터셋입니다.",
        "features": """- **해상도 및 규모**: ImageNet(`wild horse`, `zebra`)에서 수집해 **256×256 픽셀**로 전처리된 말 939장 + 얼룩말 1,177장 (train:test 약 9:1, 말:얼룩말 약 45:55).
- **비지도 스타일 이전**: 입력(말)에 대한 쌍을 이루는 정답(동일 구도의 얼룩말)이 존재하지 않으므로 unpaired 이미지 변환(CycleGAN)에 활용합니다.
- **편집 참고**: 목차(p.2)에서는 `Zebra2Horse 데이터셋`으로 표기되었으나 본문 제목은 `Horse to Zebra`이며, p.122 `데이터셋 구성` 상단에 'Pascal VOC 데이터셋은 총 20개의 클래스'라는 직전 단원 잔재 오기가 존재합니다.""",
        "structure": """| 폴더 | 내용 |
|---|---|
| `trainA` | 말(Horse) 학습용 이미지 |
| `trainB` | 얼룩말(Zebra) 학습용 이미지 |
| `testA` | 말(Horse) 테스트용 이미지 |
| `testB` | 얼룩말(Zebra) 테스트용 이미지 |""",
        "example": """- **원 데이터(Raw Data)**: 256×256 JPG 이미지(`Horse`, `Zebra`).
- **CSV 파일 예시**: Kaggle 파생 메타데이터(`image_id`, `domain`, `split`, `image_path`) 및 도메인·split 분포 시각화.""",
        "utilization": """1. **분류 모델 학습시키기**: A(Horse) vs B(Zebra) 이진 분류기 실험(Q1-1).
2. **생성 모델에 대한 이해** (`인공지능과 미래사회 p.241`).
3. **스타일 이전(Style Transfer) 모델 학습시키기**: CycleGAN 학습 및 양방향 스타일 변환 실험(Q2-1, Q2-2).""",
        "code_note": "원문 PDF(p.124)의 UC Berkeley 구 경로(`people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip`)가 404로 만료되어, 공식 현재 URL(`http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip`, 200 OK 확인)로 대체했습니다.",
        "code": """# [I08] Horse to Zebra (CycleGAN) 데이터셋 다운로드 및 리사이징 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.124-125
# 참고: 원문 URL(people.eecs.berkeley.edu)이 404로 만료되어 공식 대체 URL(efrosgans.eecs.berkeley.edu)을 적용했습니다.

import os
import urllib.request
import zipfile
import pandas as pd
import torch
import torchvision
from torchvision import datasets, transforms


def main():
    # 1. 공식 CycleGAN 서버에서 horse2zebra.zip 다운로드 (200 OK 검증됨)
    url = "http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip"
    os.makedirs("./data", exist_ok=True)
    zip_path = "./data/horse2zebra.zip"
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("./data")

    # 2. ImageFolder + 100x100 리사이징 및 저장
    resize_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    dataset_root = "./data/horse2zebra"
    dataset = datasets.ImageFolder(root=dataset_root, transform=resize_transform)
    resize_dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=8, shuffle=False)

    os.makedirs("./resized/trainA", exist_ok=True)
    os.makedirs("./resized/trainB", exist_ok=True)

    image_num = 0
    for images, _ in resize_dataloader:
        for img in images:
            torchvision.utils.save_image(img, f"./resized/trainA/{image_num}.jpg", normalize=True)
            image_num += 1
        if image_num >= 40:
            break


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "09_korean_food_image",
        "num": "09",
        "title_kr": "한국 음식 이미지 데이터셋",
        "title_en": "Korean Food Image Dataset",
        "pages": "p.126–132",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "jpg 파일",
        "category": "이미지",
        "purpose": "이미지 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=79",
        "url_pdf": "https://aihub.or.kr/aidata/13594",
        "url_status": "유효 (200 OK, AI Hub 공식 페이지로 자동 리다이렉트)",
        "textbook": "없음",
        "summary": "한국과학기술원에서 구축한 한국 음식 150종(약 15만 장) 이미지 데이터셋으로, 본 자료집에서는 교육용으로 12개 음식 클래스를 선별·축소해 활용합니다.",
        "features": """- **원본 규모**: 27종 음식 대분류(구이, 김치, 떡, 면, 밥 등)에 속하는 총 **150종**, 종목별 약 1,000장 = 총 **150,000장**.
- **원본 데이터와의 차이**: 학생 실습 편의를 위해 12개 클래스만 선정하고, 각 클래스당 **학습 500장 + 테스트 100장 = 600장(총 7,200장)**, 최대 해상도 **256×256** 이하로 축소 가공하는 코드를 제공합니다.""",
        "structure": """| 번호 | 음식 클래스 (축소 12종) | 번호 | 음식 클래스 |
|---|---|---|---|
| ① | 삼겹살 | ⑦ | 육개장 |
| ② | 깍두기 | ⑧ | 가지볶음 |
| ③ | 꿀떡 | ⑨ | 만두 |
| ④ | 라면 | ⑩ | 도토리묵 |
| ⑤ | 주먹밥 | ⑪ | 떡볶이 |
| ⑥ | 보쌈 | ⑫ | 식혜 |""",
        "example": """- **원 데이터(Raw Data)**: `train/`, `test/` 아래 12개 한글 음식명 폴더별로 `{음식명}_{번호}.jpg` 형태로 저장됩니다.""",
        "utilization": """1. **온전한 학습을 통해 한국 음식 이미지 데이터셋 분류하기**.
2. **부분 학습을 통해 한국 음식 이미지 데이터셋 분류하기**.""",
        "code_note": "AI Hub 회원가입 및 승인 후 `kfood.zip`을 내려받아야 실행할 수 있으며, 원문 코드(p.129~132)는 한글 파일 경로 처리를 위해 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` 우회 함수를 포함하고 있습니다.",
        "code": """# [I09] 한국 음식 이미지 데이터셋 압축 해제, 12종 추출 및 256px 리사이징 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.129-132
# 주의: AI Hub(https://aihub.or.kr/aidata/13594)에서 사전 신청 후 kfood.zip을 내려받아야 합니다.

import glob
import os
import zipfile
import cv2
import numpy as np


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(e)
        return False


def main():
    category_list = ["구이", "김치", "떡", "면", "밥", "찜", "국", "나물", "만두", "무침", "볶음", "음청류"]
    food_list = ["삼겹살", "깍두기", "꿀떡", "라면", "주먹밥", "보쌈", "육개장", "가지볶음", "만두", "도토리묵", "떡볶이", "식혜"]

    zips = zipfile.ZipFile("./한국 음식 이미지/kfood.zip")
    directory = "./한국 음식 이미지/"
    createFolder("./한국 음식 이미지/images")

    zipInfo = zips.infolist()
    for member in zipInfo:
        member.filename = member.filename.encode("cp437").decode("euc-kr")
        category = member.filename.split(".")[0]
        if category in category_list:
            i = category_list.index(category)
            zips.extract(member, directory)

            cate_zips = zipfile.ZipFile(f"{directory}{category_list[i]}.zip")
            for fileName in cate_zips.namelist():
                if food_list[i] in fileName:
                    cate_zips.extract(fileName, f"./한국 음식 이미지/images/{food_list[i]}")

            cnt = 0
            images = glob.glob(f"./한국 음식 이미지/images/{category_list[i]}/{food_list[i]}/*.jpg")
            createFolder("./train")
            createFolder("./test")
            createFolder(f"./train/{food_list[i]}")
            createFolder(f"./test/{food_list[i]}")

            for fname in images:
                img = imread(fname)
                if img is None:
                    continue
                cnt += 1
                height, width, _ = img.shape
                if width > height:
                    height = int(height * 256 / width)
                    width = 256
                else:
                    width = int(width * 256 / height)
                    height = 256
                img = cv2.resize(img, (width, height))

                if cnt <= 500:
                    imwrite(f"./train/{food_list[i]}/{food_list[i]}_{cnt}.jpg", img)
                elif cnt <= 600:
                    imwrite(f"./test/{food_list[i]}/{food_list[i]}_{cnt - 500}.jpg", img)
                else:
                    break


if __name__ == "__main__":
    main()
""",
    },
    {
        "slug": "10_korean_font_image",
        "num": "10",
        "title_kr": "한국어 글자체 이미지 데이터셋",
        "title_en": "Korean font image dataset",
        "pages": "p.133–139",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "png 파일 (+ json 라벨 및 어노테이션)",
        "category": "이미지",
        "purpose": "이미지 분류 / 한글 광학 문자 인식(OCR) / 지도학습",
        "url": "https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81",
        "url_pdf": "https://aihub.or.kr/aidata/133",
        "url_status": "유효 (200 OK, AI Hub 공식 페이지로 자동 리다이렉트)",
        "textbook": "없음",
        "summary": "현대 한글 11,172자 전체를 50종 폰트별 인쇄체(280만자), 손글씨(370만자), 실사 글자(간판/표지판) 이미지와 JSON 어노테이션으로 구축한 한국어 OCR 데이터셋입니다.",
        "features": """- **출처별 구성**: ① 손글씨(370만 자) ② 인쇄체(50종 폰트, 280만 자) ③ 실사 글자(간판·이정표 등, 바운딩 박스 `annotations[].bbox` 포함) — 실사 포함 이미지 **10만 장**.
- **단위별 구성**: ① 음절(글자, 11,172자 단위) ② 어절(단어, 최대 14,000자 약 6,000개) ③ 문장(최대 30,000자 약 1,000개).""",
        "structure": """| JSON 최상위 키 | 포함 정보 |
|---|---|
| `info` | 데이터셋 이름, 설명, 버전, 생성일 |
| `images` | 이미지 ID, 가로/세로 크기, 파일명, 라이선스 |
| `annotations` | 어노테이션 ID, `image_id`, `text`(정답 글자), 실사 글자 시 `bbox` 좌표 |
| `licenses` | 라이선스 이름 및 URL |""",
        "example": """- **원 데이터(Raw Data)**: PNG 글자 이미지와 JSON 어노테이션 파일(`손글씨/인쇄체` 및 `실사 글자` JSON 구조 비교).""",
        "utilization": """1. **데이터셋을 구성하는 속성들의 전처리(증강: Augmentation) 해보기**.
2. **타겟 데이터 결정하기**: JSON 어노테이션 기준으로 음절/어절/문장 및 출처 필터링 후 7:3 또는 8:2 분할.
3. **AI 기반 광학 글자인식(OCR) 구현하기**: 한글 인식 분류 모델 학습 및 테스트(Q1-1, Q1-2).""",
        "code_note": "AI Hub에서 데이터를 내려받아 JSON 메타데이터를 파싱하고 `ImageFolder`와 `DataLoader`로 적재하는 원문 코드(p.136~138)입니다.",
        "code": """# [I10] 한국어 글자체 이미지 데이터셋 JSON 파싱 및 PyTorch DataLoader 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.136-138
# 주의: AI Hub(https://aihub.or.kr/aidata/133)에서 데이터를 사전 다운로드한 뒤 경로를 지정하세요.

import json
import torch
from PIL import Image
from torchvision import datasets, transforms


def main():
    # 1. JSON 파일(라벨 및 어노테이션 데이터) 읽기
    json_path = "./data/annotation.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        print(json.dumps(json_data, ensure_ascii=False)[:500])

    # 2. 이미지 전처리 및 ImageFolder / DataLoader 구성
    korean_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    data_root = "./data/images"
    train_dataset = datasets.ImageFolder(root=data_root, transform=korean_transform)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
    )

    images, labels = next(iter(train_dataloader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
""",
    },
]


def render_readme(it: dict) -> str:
    return f"""# {it['num']}. {it['title_kr']} ({it['title_en']})

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), {it['pages']}

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | {it['title_kr']} ({it['title_en']}) |
| **카테고리** | {it['category']} |
| **데이터셋 목적** | {it['purpose']} |
| **난이도** | {it['difficulty']} |
| **흥미도** | {it['interest']} |
| **파일 형태** | {it['format']} |
| **데이터셋 링크** | {it['url']} |
| **원문 PDF 표기 링크** | {it['url_pdf']} |
| **링크 검증 상태 (2026-11)** | {it['url_status']} |
| **교과서 연계** | {it['textbook']} |

{it['summary']}

---

## 2. 데이터셋 특징

{it['features']}

---

## 3. 데이터셋 구성

{it['structure']}

---

## 4. 데이터셋 예시

{it['example']}

---

## 5. 데이터셋 활용

{it['utilization']}

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: {it['code_note']}

```python
{it['code'].strip()}
```
"""


def main() -> None:
    index_rows = []
    for it in ITEMS:
        d = BASE / it["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(render_readme(it), encoding="utf-8")
        (d / "prepare_dataset.py").write_text(it["code"], encoding="utf-8")
        index_rows.append(
            f"| {it['num']} | [{it['title_kr']}](./{it['slug']}/README.md) | {it['purpose']} | {it['difficulty']} | {it['interest']} | {it['format']} | [링크]({it['url']}) | [코드](./{it['slug']}/prepare_dataset.py) |"
        )

    section_readme = f"""# Ⅱ. 이미지 데이터셋 (Image Datasets)

기초 이미지 분류(MNIST, Fashion MNIST, CIFAR10, Dogs vs. Cats, SVHN), 비지도·고해상도 표현 학습(STL-10), 객체 검출(PASCAL VOC), 스타일 이전(Horse to Zebra), 국내 특화 데이터(한국 음식, 한국어 글자체 OCR) 총 10개 데이터셋을 수록합니다.

> **참고**: 본문 순서 기준으로 `02 Fashion MNIST(p.081)`와 `03 CIFAR10(p.087)`을 배열했습니다(원문 p.2 목차 표기와 순서 반대). 또한 원문 링크가 만료(404)된 `01 MNIST` 및 `08 Horse to Zebra`는 동일 데이터셋의 공식 대체 링크로 업데이트했습니다.

| 번호 | 데이터셋 명 | 데이터셋 목적 | 난이도 | 흥미도 | 형태 | 공식 링크 | 예시 코드 |
|---|---|---|---|---|---|---|---|
{chr(10).join(index_rows)}
"""
    (BASE / "README.md").write_text(section_readme, encoding="utf-8")
    print(f"Generated {len(ITEMS)} datasets in {BASE}")


if __name__ == "__main__":
    main()
