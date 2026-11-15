# 06. STL-10 데이터셋 (STL-10 Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.106–112

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | STL-10 데이터셋 (STL-10 Dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(지도학습) 및 비지도 표현 학습(Unsupervised Learning) |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | mat/bin 파일 (추가로 png 등 저장 가능) |
| **데이터셋 링크** | https://cs.stanford.edu/~acoates/stl10/ |
| **원문 PDF 표기 링크** | https://cs.stanford.edu/~acoates/stl10/ |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Stanford 공식 페이지) |
| **교과서 연계** | 없음 |

ImageNet에서 추출한 96×96 고해상도 컬러 이미지 데이터셋으로, 소량의 라벨 데이터(지도학습)와 대규모 비라벨 이미지 10만 장(비지도학습)을 함께 제공합니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: **96×96 픽셀 3채널 RGB** (CIFAR-10의 32×32보다 훨씬 높은 해상도).
- **규모**:
  - **지도학습용(10클래스)**: 클래스별 학습 500장(총 5,000장) + 테스트 800장(총 8,000장).
  - **비지도학습용(`unlabeled`)**: 라벨 없는 **100,000장** (10클래스 외 곰, 토끼, 기차, 버스 등 다양한 객체 포함).

---

## 3. 데이터셋 구성

| 구분 | 클래스 / 수량 | 설명 |
|---|---|---|
| 지도학습 10클래스 | ① airplane ② bird ③ car ④ cat ⑤ deer ⑥ dog ⑦ horse ⑧ monkey ⑨ ship ⑩ truck | train 5,000장 / test 8,000장 (96×96 RGB) |
| 비지도학습 (`unlabeled`) | 라벨 없음 (다양한 동물·탈것 포함) | 100,000장 (96×96 RGB) |

---

## 4. 데이터셋 예시

- **1. 지도학습용 이미지**: 10개 클래스별 실사 이미지.
- **2. 비지도학습용 이미지**: 코끼리, 사슴, 선박, 기차, 이구아나 등 폭넓은 분포의 라벨 없는 이미지.

---

## 5. 데이터셋 활용

1. **데이터셋 분리**: `StratifiedShuffleSplit`을 이용해 `test_ds0`(8,000장)을 계층적 샘플링으로 검증(validation)과 테스트(test)로 8:2 분할(Q1-1).
2. **분류 모델 학습 및 정량 평가(Quantitative)** (Q2-1~Q3-3).
3. **(심화) 비지도 학습 개념 이해하기**: 군집화(Clustering) 및 `unlabeled` 데이터셋 로드 비교(Q4-1~Q4-2).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.109~112)는 `torchvision.datasets.STL10` 다운로드, `StratifiedShuffleSplit` 검증 분할, 그리고 `torchvision.utils.save_image`를 이용한 라벨별 PNG 추출 코드를 제공합니다.

```python
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
```
