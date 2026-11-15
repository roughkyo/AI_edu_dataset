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
