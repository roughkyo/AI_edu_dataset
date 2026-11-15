# 08. Horse to Zebra 데이터셋 (Horse2zebra dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.121–125

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Horse to Zebra 데이터셋 (Horse2zebra dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 스타일 이전(Style Transfer) / 비지도학습(Unsupervised Learning) |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | jpg+csv 파일 |
| **데이터셋 링크** | http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/ |
| **원문 PDF 표기 링크** | https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/ (404 만료로 공식 대체) |
| **링크 검증 상태 (2026-11)** | 대체 완료 (200 OK, UC Berkeley CycleGAN 공식 서버 / 직접 ZIP: http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip) |
| **교과서 연계** | 인공지능과 미래사회 p.241 (스타일 트랜스퍼) |

짝지어지지 않은(unpaired) 말(Horse)과 얼룩말(Zebra) 이미지를 활용해 CycleGAN 등의 이미지 스타일 이전 모델을 학습시키는 비지도학습 데이터셋입니다.

---

## 2. 데이터셋 특징

- **해상도 및 규모**: ImageNet(`wild horse`, `zebra`)에서 수집해 **256×256 픽셀**로 전처리된 말 939장 + 얼룩말 1,177장 (train:test 약 9:1, 말:얼룩말 약 45:55).
- **비지도 스타일 이전**: 입력(말)에 대한 쌍을 이루는 정답(동일 구도의 얼룩말)이 존재하지 않으므로 unpaired 이미지 변환(CycleGAN)에 활용합니다.
- **편집 참고**: 목차(p.2)에서는 `Zebra2Horse 데이터셋`으로 표기되었으나 본문 제목은 `Horse to Zebra`이며, p.122 `데이터셋 구성` 상단에 'Pascal VOC 데이터셋은 총 20개의 클래스'라는 직전 단원 잔재 오기가 존재합니다.

---

## 3. 데이터셋 구성

| 폴더 | 내용 |
|---|---|
| `trainA` | 말(Horse) 학습용 이미지 |
| `trainB` | 얼룩말(Zebra) 학습용 이미지 |
| `testA` | 말(Horse) 테스트용 이미지 |
| `testB` | 얼룩말(Zebra) 테스트용 이미지 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 256×256 JPG 이미지(`Horse`, `Zebra`).
- **CSV 파일 예시**: Kaggle 파생 메타데이터(`image_id`, `domain`, `split`, `image_path`) 및 도메인·split 분포 시각화.

---

## 5. 데이터셋 활용

1. **분류 모델 학습시키기**: A(Horse) vs B(Zebra) 이진 분류기 실험(Q1-1).
2. **생성 모델에 대한 이해** (`인공지능과 미래사회 p.241`).
3. **스타일 이전(Style Transfer) 모델 학습시키기**: CycleGAN 학습 및 양방향 스타일 변환 실험(Q2-1, Q2-2).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 PDF(p.124)의 UC Berkeley 구 경로(`people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip`)가 404로 만료되어, 공식 현재 URL(`http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/horse2zebra.zip`, 200 OK 확인)로 대체했습니다.

```python
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
```
