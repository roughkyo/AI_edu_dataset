# 05. SVHN 데이터셋 (Street View House Numbers (SVHN))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.099–105

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | SVHN 데이터셋 (Street View House Numbers (SVHN)) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | png+mat 파일 |
| **데이터셋 링크** | http://ufldl.stanford.edu/housenumbers/ |
| **원문 PDF 표기 링크** | http://ufldl.stanford.edu/housenumbers |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Stanford UFLDL 공식 페이지) |
| **교과서 연계** | 없음 |

Google Street View에서 수집한 실제 건물 번호판·표지판 숫자(0~9)를 담은 32×32 RGB 컬러 이미지 데이터셋으로, MNIST의 실사 심화 버전입니다.

---

## 2. 데이터셋 특징

- **해상도 및 채널**: Format 2 기준 **32×32 픽셀 RGB 3채널** 이미지(한 사진 안에 인접 숫자가 일부 포함되기도 함).
- **규모**: 학습용 73,257개, 테스트용 26,032개, 추가 난이도 낮은 데이터(`extra`) 531,131개.
- **원본 포맷 2종**: Format 1(원본 해상도 PNG + `digitStruct.mat` 바운딩 박스), Format 2(32×32 크롭된 `train_32x32.mat`, `test_32x32.mat` — `torchvision` 기본 지원).

---

## 3. 데이터셋 구성

| 라벨 | 대상 | 비고 |
|---|---|---|
| `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` (10클래스) | 실제 거리 번호판 숫자 | Format 2 기준 32×32 RGB |

---

## 4. 데이터셋 예시

- **Format 1**: 건물 전체 표지판 이미지와 숫자 영역(Bounding Box).
- **Format 2**: 32×32로 균일하게 잘라낸 숫자 타일 이미지(`X`, `y` 행렬).

---

## 5. 데이터셋 활용

1. **학습데이터, 테스트 데이터를 위한 DataLoader 만들기**: `torchvision.transforms`의 다양한 데이터 증강 실험(Q1-1).
2. **분류 모델 정량적 성능 파악 (Quantitative)**: CNN 설계, 학습, 모델 저장 및 테스트 정확도 산출(Q2-1~Q3-2).
3. **분류 모델 정성적 성능 파악 (Qualitative)**: 예측 결과와 정답 라벨·이미지 직접 비교 시각화(Q4-1~Q4-3).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.102~104)는 `torchvision.datasets.SVHN`으로 Format 2(`.mat`)를 자동 다운로드하고 `DataLoader` 학습/평가 루프 템플릿을 제공합니다(오타 `transfrom`→`transform` 수정).

```python
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
```
