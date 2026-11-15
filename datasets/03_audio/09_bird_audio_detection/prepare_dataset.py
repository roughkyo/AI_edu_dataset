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
