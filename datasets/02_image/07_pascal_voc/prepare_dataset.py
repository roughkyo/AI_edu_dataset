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
