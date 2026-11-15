# [I08] Horse to Zebra (CycleGAN) 데이터셋 다운로드 및 리사이징 예제
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
