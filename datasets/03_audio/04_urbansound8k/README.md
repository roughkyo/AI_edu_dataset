# 04. URBAN SOUND 데이터셋 (URBAN SOUND Dataset (UrbanSound8K))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.161–167

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | URBAN SOUND 데이터셋 (URBAN SOUND Dataset (UrbanSound8K)) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 음성 분류(Audio Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://urbansounddataset.weebly.com/urbansound8k.html |
| **원문 PDF 표기 링크** | https://urbansounddataset.weebly.com/urbansound8k.html |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, NYU UrbanSound8K 공식 페이지) |
| **교과서 연계** | 없음 |

도시 환경에서 발생하는 10가지 소음(에어컨, 경적, 개 짖는 소리, 드릴, 사이렌 등)을 담은 8,732개(약 5.6GB, 4초 이내) 오디오 분류 데이터셋입니다.

---

## 2. 데이터셋 특징

- **규모**: 10개 클래스, 10개 fold로 구성된 **8,732개** `.wav` 파일(44.1/48kHz, 총 5.6GB).
- **핵심 메타데이터**: `UrbanSound8K.csv`에 `slice_file_name`, `fsID`, `start`, `end`, `salience`(1: foreground, 2: background), `fold`(1~10), `classID`, `class`가 기록되어 있습니다.

---

## 3. 데이터셋 구성

| classID | 클래스명 | 설명 |
|---|---|---|
| 0~9 (10종) | `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music` | 도시 소음 10종 (4초 이하) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 10개 fold 폴더(`fold1`~`fold10`)별 `.wav` 파일과 `UrbanSound8K.csv`.
- **시각화 데이터**: 160,000 샘플 길이로 제로패딩 통일 후 변환한 스펙트로그램 이미지.

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 160,000 샘플 고정 길이 패딩/트림 및 스테레오→모노 평균화.
2. **음성 데이터 시각화 해보기**: 스펙트로그램 이미지 추출.
3. **전체 학습을 통해 음성 분류하기**: fold 기반 train(`1~9`)/test(`10`) 분할 학습.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.164~167)는 Colab에서 `opendatasets` + Kaggle API(`chrisfilo/urbansound8k`)로 데이터를 내려받아 `UrbanSoundDataset` 클래스로 길이 통일(160,000) 후 WAV 및 JPG 스펙트로그램으로 저장합니다.

```python
# [V04] UrbanSound8K 데이터셋 PyTorch 전처리, 길이 통일 및 스펙트로그램 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.164-167
# [Colab/Shell] !pip install torchaudio opendatasets --upgrade --quiet

import os
import matplotlib.pyplot as plt
import opendatasets as od
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


class UrbanSoundDataset(Dataset):
    def __init__(self, csv_path, file_path, folderList):
        csvData = pd.read_csv(csv_path)
        self.file_names = []
        self.labels = []
        self.folders = []
        for i in range(len(csvData)):
            if csvData.iloc[i, 5] in folderList:
                self.file_names.append(csvData.iloc[i, 0])
                self.labels.append(csvData.iloc[i, 6])
                self.folders.append(csvData.iloc[i, 5])
        self.file_path = file_path

    def __getitem__(self, index):
        path = f"{self.file_path}/fold{self.folders[index]}/{self.file_names[index]}"
        sound, sampling_rate = torchaudio.load(path)
        # 스테레오/다채널 -> 모노 평균
        sound = torch.mean(sound, dim=0, keepdim=True)
        sound = torch.transpose(sound, 0, 1)

        # 160,000 샘플로 제로패딩 또는 자르기
        tempData = torch.zeros([160000, 1])
        if sound.numel() < 160000:
            tempData[: sound.numel(), 0] = sound.squeeze(1)
        else:
            tempData[:, 0] = sound[:160000, 0]
        return tempData, self.labels[index], sampling_rate

    def __len__(self):
        return len(self.file_names)


def main():
    # Kaggle 인증 후 다운로드 (또는 공식 사이트 https://urbansounddataset.weebly.com/urbansound8k.html 활용)
    # dataset_url = "https://www.kaggle.com/chrisfilo/urbansound8k"
    # od.download(dataset_url)

    csv_path = "./urbansound8k/UrbanSound8K.csv"
    file_path = "./urbansound8k"

    train_set = UrbanSoundDataset(csv_path, file_path, range(1, 10))
    test_set = UrbanSoundDataset(csv_path, file_path, [10])
    print(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=512, win_length=200, hop_length=80),
        AT.AmplitudeToDB(),
    )

    labels = [
        "air_conditioner", "car_horn", "children_playing", "dog_bark", "drilling",
        "engine_idling", "gun_shot", "jackhammer", "siren", "street_music",
    ]
    cnts = torch.zeros(10, dtype=int)
    for lbl in labels:
        createFolder(f"./urbansound/{lbl}")

    for sound, label, sampling_rate in train_loader:
        s = sound[0]
        lbl_idx = label.item()
        sr = sampling_rate.item()
        sf.write(f"./urbansound/{labels[lbl_idx]}/{labels[lbl_idx]}_{cnts[lbl_idx]}.wav", s.numpy(), sr)
        spec = spectrogram(s.transpose(0, 1))
        plt.pcolor(spec[0].numpy())
        plt.savefig(f"./urbansound/{labels[lbl_idx]}/{labels[lbl_idx]}_{cnts[lbl_idx]}.jpg")
        plt.clf()
        plt.close()
        cnts[lbl_idx] += 1


if __name__ == "__main__":
    main()
```
