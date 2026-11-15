# [V05] Speech Commands 데이터셋 클래스별 샘플 추출 및 스펙트로그램 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.171-172

import os
import matplotlib.pyplot as plt
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
    # 1. torchaudio로 SPEECHCOMMANDS 다운로드
    audios = torchaudio.datasets.SPEECHCOMMANDS("./", download=True)
    data_loader = torch.utils.data.DataLoader(audios, batch_size=1, shuffle=True)

    # 2. 16kHz 기반 Spectrogram 변환기
    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=512, win_length=400, hop_length=160),
        AT.AmplitudeToDB(),
    )

    # 3. 클래스별 최대 10개씩 WAV 및 스펙트로그램 저장
    dicts = {}
    for data in data_loader:
        audio = data[0][0]
        y = data[2][0]

        if y in dicts:
            dicts[y] += 1
        else:
            dicts[y] = 0
            createFolder(f"./{y}")

        if dicts[y] < 10:
            sf.write(
                f"./{y}/{y}_{dicts[y]}.wav",
                torch.transpose(audio, 0, 1).numpy(),
                data[1].item(),
            )
            spec = spectrogram(audio)
            plt.pcolor(spec[0].numpy())
            plt.savefig(f"./{y}/{y}_{dicts[y]}.jpg")
            plt.clf()
            plt.close()


if __name__ == "__main__":
    main()
