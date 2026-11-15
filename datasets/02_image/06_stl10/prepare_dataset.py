# [I06] STL-10 데이터셋 로드, 계층적 분할 및 라벨별 PNG 저장 예제
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
