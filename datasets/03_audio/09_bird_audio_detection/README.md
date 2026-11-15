# 09. Bird Audio Detection challenge 데이터셋 (Bird Audio Detection challenge dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.190–195

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Bird Audio Detection challenge 데이터셋 (Bird Audio Detection challenge dataset) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 이진 분류(새소리 유무) / 지도학습 / 도메인 일반화(Domain Generalization) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | wav 파일 (+ csv 메타데이터) |
| **데이터셋 링크** | https://dcase.community/challenge2018/task-bird-audio-detection |
| **원문 PDF 표기 링크** | http://dcase.community/challenge2018/task-bird-audio-detection |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, DCASE 공식 페이지로 리다이렉트) |
| **교과서 연계** | 없음 |

서로 다른 지역·장비·환경에서 수집된 6개 음원 세트를 이용해 새소리 존재 여부(`hasbird`)를 판별하고 새로운 녹음 환경으로 일반화(Domain Adaptation)하는 DCASE 챌린지 데이터셋입니다.

---

## 2. 데이터셋 특징

- **도메인 일반화 전략**: 동일 데이터셋 내부 분할이 아닌 서로 다른 환경의 3개 개발 세트(train)와 3개 평가 세트(validation/test)로 교차 검증을 권장합니다.
- **규모**:
  - Development (3종, 총 **35,690개**): `freefield1010`(7,690개, 5.8GB), `warblrb10k`(8,000개, 4.3GB), `BirdVox-DCASE-20k`(20,000개, 15.4GB).
  - Evaluation (3종, 총 **12,620개**): `warblrb10k eval`(2,000개, 1.3GB), `Chernobyl`(6,620개, 5.3GB), `PolandNFC`(4,000개, 2.3GB).
- **편집 참고**: p.190 제목에 `"데이터셋 데이터셋"` 중복 오타가 있습니다.

---

## 3. 데이터셋 구성

| CSV 열 이름 | 설명 |
|---|---|
| `itemid` | 음성 파일 일련번호 (파일명) |
| `datasetid` | 소속 데이터셋 종류 (`warblrb10k` 등) |
| `hasbird` | 새 존재 여부 이진 라벨 (`0`: 없음, `1`: 있음) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 6개 데이터셋 압축 ZIP 파일 및 메타 CSV.
- **시각화 데이터**: 특정 음원의 스펙트로그램 예시.

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기** 및 **시각화 해보기**.
2. **전체 학습을 통해 음성 분류 / 도메인 일반화 적용해보기**: train/validation 분할 및 타 도메인 교차 검증 비교(Q3-1, Q3-2).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.193~195)는 Colab 환경에서 `warblrb10k_public_wav.zip`과 `warblrb10k_public_metadata_2018.csv`를 업로드한 뒤 `BirdSoundDataset`으로 로드하여 `Presence`/`Absence` 폴더별 WAV 및 스펙트로그램 JPG를 저장합니다.

```python
# [V09] Bird Audio Detection (warblrb10k) PyTorch 전처리 및 스펙트로그램 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.193-195
# [Colab/Shell] !pip install torchaudio
# [Colab/Shell] !unzip -uq "/content/warblrb10k_public_wav.zip" -d "/content/wav"

import os
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT
from torch.utils.data import DataLoader, Dataset


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


class BirdSoundDataset(Dataset):
    def __init__(self, csv_path, file_path):
        csvData = pd.read_csv(csv_path)
        self.file_names = [csvData.iloc[i, 0] for i in range(len(csvData))]
        self.labels = [csvData.iloc[i, 2] for i in range(len(csvData))]
        self.file_path = file_path

    def __getitem__(self, index):
        path = f"{self.file_path}/{self.file_names[index]}.wav"
        soundData, sampling_rate = torchaudio.load(path)
        sound = torch.transpose(soundData, 1, 0)
        return sound, self.labels[index], sampling_rate

    def __len__(self):
        return len(self.file_names)


def main():
    csv_path = "./warblrb10k_public_metadata_2018.csv"
    file_path = "./wav"

    train_set = BirdSoundDataset(csv_path, file_path)
    print(f"Train set size: {len(train_set)}")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=512, win_length=200, hop_length=80),
        AT.AmplitudeToDB(),
    )

    labels = ["Absence", "Presence"]  # 0: Absence, 1: Presence
    cnts = torch.zeros(2, dtype=int)
    for lbl in labels:
        createFolder(f"./birdsound/{lbl}")

    for i, (sound, label, sampling_rate) in enumerate(train_loader):
        s = sound[0]
        lbl_idx = label.item()
        sr = sampling_rate.item()
        sf.write(f"./birdsound/{labels[lbl_idx]}/{labels[lbl_idx]}_{cnts[lbl_idx]}.wav", s.numpy(), sr)
        if i == 0:
            spec = spectrogram(s.transpose(0, 1))
            plt.pcolor(spec[0].numpy())
            plt.savefig(f"./birdsound/{labels[lbl_idx]}/{labels[lbl_idx]}_{cnts[lbl_idx]}.jpg")
            plt.clf()
            plt.close()
        cnts[lbl_idx] += 1


if __name__ == "__main__":
    main()
```
