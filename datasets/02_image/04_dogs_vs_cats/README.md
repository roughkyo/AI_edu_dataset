# 04. Dogs vs. Cats 데이터셋 (Dogs vs. Cats Datasets)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.092–098

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Dogs vs. Cats 데이터셋 (Dogs vs. Cats Datasets) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(이진 분류) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | jpg 파일 |
| **데이터셋 링크** | https://www.kaggle.com/c/dogs-vs-cats/overview |
| **원문 PDF 표기 링크** | https://www.kaggle.com/c/dogs-vs-cats/overview |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle Competition 공식 페이지) |
| **교과서 연계** | 없음 |

가변 해상도의 RGB 컬러 이미지에서 개(dog)와 고양이(cat)를 구분하는 대표적 이진 이미지 분류 데이터셋입니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: 이미지마다 가로·세로 크기가 다르며 RGB 3채널 컬러(`0~255`)로 구성됩니다.
- **규모**: 학습용 `train.zip` 25,000장 + 테스트용 `test1.zip` 12,500장 = 총 **37,500장**.
- **전처리 핵심**: `test1.zip`에는 정답 파일명이 없으므로 `train.zip`(개 12,500장, 고양이 12,500장)을 학습 22,000장 / 테스트(검증) 3,000장으로 분할하고 `torchvision.transforms`로 크기를 통일합니다.

---

## 3. 데이터셋 구성

| 라벨 | 클래스 | 설명 |
|---|---|---|
| `0` | `dog` | 개 이미지 (학습 12,500장 중 11,000장 train / 1,500장 test 분할) |
| `1` | `cat` | 고양이 이미지 (학습 12,500장 중 11,000장 train / 1,500장 test 분할) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 다양한 비율의 개·고양이 실사 JPG 이미지(`dog.0.jpg`~`dog.12499.jpg`, `cat.0.jpg`~`cat.12499.jpg`).

---

## 5. 데이터셋 활용

1. **이미지 시각화하기**: 전처리된 `DataLoader` 배치를 4×6 바둑판 그리드로 시각화.
2. **학습/테스트 전처리**: `Resize(256, 256)`, `RandomCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`.
3. **학습·테스트 데이터 분류(7:3, 8:2)** 및 **분류 네트워크 설계하기**.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.095)의 Google Drive URL(`1scy6yMrhaOLXjhrVbpD70zxk5n6SafSD`)은 만료(404)되었습니다. Kaggle Competition 페이지에서 `train.zip`을 내려받아 `./train/`에 압축 해제한 뒤 실행하세요.

```python
# [I04] Dogs vs. Cats 데이터셋 PyTorch CustomDataset 및 전처리 예제
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
```
