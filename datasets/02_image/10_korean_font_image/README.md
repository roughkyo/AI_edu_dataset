# 10. 한국어 글자체 이미지 데이터셋 (Korean font image dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.133–139

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 한국어 글자체 이미지 데이터셋 (Korean font image dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류 / 한글 광학 문자 인식(OCR) / 지도학습 |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | png 파일 (+ json 라벨 및 어노테이션) |
| **데이터셋 링크** | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81 |
| **원문 PDF 표기 링크** | https://aihub.or.kr/aidata/133 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, AI Hub 공식 페이지로 자동 리다이렉트) |
| **교과서 연계** | 없음 |

현대 한글 11,172자 전체를 50종 폰트별 인쇄체(280만자), 손글씨(370만자), 실사 글자(간판/표지판) 이미지와 JSON 어노테이션으로 구축한 한국어 OCR 데이터셋입니다.

---

## 2. 데이터셋 특징

- **출처별 구성**: ① 손글씨(370만 자) ② 인쇄체(50종 폰트, 280만 자) ③ 실사 글자(간판·이정표 등, 바운딩 박스 `annotations[].bbox` 포함) — 실사 포함 이미지 **10만 장**.
- **단위별 구성**: ① 음절(글자, 11,172자 단위) ② 어절(단어, 최대 14,000자 약 6,000개) ③ 문장(최대 30,000자 약 1,000개).

---

## 3. 데이터셋 구성

| JSON 최상위 키 | 포함 정보 |
|---|---|
| `info` | 데이터셋 이름, 설명, 버전, 생성일 |
| `images` | 이미지 ID, 가로/세로 크기, 파일명, 라이선스 |
| `annotations` | 어노테이션 ID, `image_id`, `text`(정답 글자), 실사 글자 시 `bbox` 좌표 |
| `licenses` | 라이선스 이름 및 URL |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: PNG 글자 이미지와 JSON 어노테이션 파일(`손글씨/인쇄체` 및 `실사 글자` JSON 구조 비교).

---

## 5. 데이터셋 활용

1. **데이터셋을 구성하는 속성들의 전처리(증강: Augmentation) 해보기**.
2. **타겟 데이터 결정하기**: JSON 어노테이션 기준으로 음절/어절/문장 및 출처 필터링 후 7:3 또는 8:2 분할.
3. **AI 기반 광학 글자인식(OCR) 구현하기**: 한글 인식 분류 모델 학습 및 테스트(Q1-1, Q1-2).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: AI Hub에서 데이터를 내려받아 JSON 메타데이터를 파싱하고 `ImageFolder`와 `DataLoader`로 적재하는 원문 코드(p.136~138)입니다.

```python
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
```
