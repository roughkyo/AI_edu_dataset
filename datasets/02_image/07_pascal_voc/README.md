# 07. PASCAL VOC 데이터셋 (Pascal VOC Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.113–120

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | PASCAL VOC 데이터셋 (Pascal VOC Dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 객체 검출(Object Detection) / 지도학습 |
| **난이도** | ★★★★★ (5/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | jpg 파일 (+ xml 어노테이션) |
| **데이터셋 링크** | https://pjreddie.com/projects/pascal-voc-dataset-mirror/ |
| **원문 PDF 표기 링크** | https://pjreddie.com/projects/pascal-voc-dataset-mirror/ |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, pjreddie 공식 미러 및 tar 파일 정상 확인) |
| **교과서 연계** | 없음 |

하나의 이미지 안에 여러 클래스의 객체가 동시에 등장하며, 바운딩 박스(Bounding Box)와 세그멘테이션 마스크를 제공하는 대표적 객체 검출 벤치마크입니다.

---

## 2. 데이터셋 특징

- **20개 객체 클래스**와 각 객체의 위치 좌표(`xmin, ymin, xmax, ymax` XML 어노테이션)를 포함합니다.
- **규모**:
  - **VOC2007**: 객체 12,608개 포함 이미지 5,011장 (예제 코드 기준 `train` 2,501장 + `val` 2,510장 / `test` 4,952장).
  - **VOC2012**: 객체 27,450개 포함 이미지 11,540장.
- **전처리 핵심**: 이미지마다 해상도와 비율이 다르므로 `albumentations`의 `LongestMaxSize` + `PadIfNeeded`를 활용해 **바운딩 박스 좌표까지 함께 보정**하며 600×600으로 통일합니다.

---

## 3. 데이터셋 구성

| 번호 범위 | 클래스 목록 (총 20개) |
|---|---|
| 0 ~ 3 | `aeroplane`, `bicycle`, `bird`, `boat` |
| 4 ~ 7 | `bottle`, `bus`, `car`, `cat` |
| 8 ~ 11 | `chair`, `cow`, `diningtable`, `dog` |
| 12 ~ 15 | `horse`, `motorbike`, `person`, `pottedplant` |
| 16 ~ 19 | `sheep`, `sofa`, `train`, `tvmonitor` |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 다양한 비율의 실사 JPG 이미지 + XML 어노테이션 + 바운딩 박스 시각화.

---

## 5. 데이터셋 활용

1. **이미지 시각화하기**: 원본 이미지 로드 확인.
2. **바운딩 박스 시각화하기**: XML에서 파싱한 다중 바운딩 박스와 클래스 라벨을 이미지 위에 그려 확인.
3. **객체 검출(Object Detection)**: 위치 추정(Localization)과 분류(Classification)를 동시에 수행하는 모델 학습 준비.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.115~120)는 Google Colab 환경 기준으로 작성되어 있으며(`!pip install -U albumentations`, `%matplotlib inline`), `VOCDetection`을 상속한 `myVOCDetection` 클래스로 XML 파싱, `albumentations` 리사이징, 바운딩 박스 시각화 및 PNG 저장을 수행합니다.

```python
# [I07] PASCAL VOC 2007 데이터셋 로드, 바운딩 박스 파싱 및 Albumentations 전처리 예제
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
```
