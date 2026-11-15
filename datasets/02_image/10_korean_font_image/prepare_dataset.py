# [I10] 한국어 글자체 이미지 데이터셋 JSON 파싱 및 PyTorch DataLoader 예제
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
