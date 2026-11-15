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
