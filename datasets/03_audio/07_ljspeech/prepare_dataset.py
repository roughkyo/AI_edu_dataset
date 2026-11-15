# [V07] LJSpeech 데이터셋 100개 샘플 추출 및 대본 CSV 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.182-183

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as AT


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def main():
    audios = torchaudio.datasets.LJSPEECH("./", download=True)
    data_loader = torch.utils.data.DataLoader(audios, batch_size=1, shuffle=True)

    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=550, win_length=550, hop_length=220),
        AT.AmplitudeToDB(),
    )

    createFolder("./LJSPEECH")
    labels = []
    speechs = []
    cnt = 0

    for data in data_loader:
        cnt += 1
        audio = data[0][0]

        sf.write(f"./LJSPEECH/{cnt}.wav", torch.transpose(audio, 0, 1).numpy(), data[1].item())
        spec = spectrogram(audio)
        plt.pcolor(spec[0].numpy())
        plt.savefig(f"./LJSPEECH/{cnt}.jpg")
        plt.clf()
        plt.close()

        labels.append(data[2][0])
        speechs.append(data[3][0])

        if cnt == 100:
            break

    pd.DataFrame(np.asarray(labels)).to_csv("./LJSPEECH/label.csv", index=False)
    pd.DataFrame(np.asarray(speechs)).to_csv("./LJSPEECH/speech.csv", index=False)


if __name__ == "__main__":
    main()
