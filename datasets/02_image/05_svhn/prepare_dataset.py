# [I05] SVHN 데이터셋 PyTorch 로드 및 학습/평가 루프 예제
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
