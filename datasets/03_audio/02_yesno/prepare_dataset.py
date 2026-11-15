# [V02] Yes/No 데이터셋 VAD 무음 제거 및 8등분 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.152-154

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
    # 1. PyTorch torchaudio를 통해 YESNO 다운로드
    yesno_data = torchaudio.datasets.YESNO("./", download=True)
    data_loader = torch.utils.data.DataLoader(yesno_data, batch_size=1, shuffle=True)

    # 2. Spectrogram 변환기 정의 (8kHz 기준)
    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=256, win_length=200, hop_length=80),
        AT.AmplitudeToDB(),
    )

    label_names = ["no", "yes"]
    cnts = [0, 0]
    createFolder("./yes")
    createFolder("./no")

    # 3. 무음 제거(앞/뒤 vad) 후 8등분 분할 저장
    for data in data_loader:
        _, _, n = data[0].shape
        trimmed = torchaudio.functional.vad(data[0][0], data[1], search_time=20)
        _, n2 = trimmed.shape
        alpha = int(n - n2)
        trimmed = trimmed[:, : max(1, n2 - alpha)]

        _, n_trim = trimmed.shape
        step = int(n_trim / 8)
        audio = trimmed

        for i in range(8):
            y = data[2][i].item()
            cnts[y] += 1
            chunk = audio[:, step * i : step * (i + 1)]
            sf.write(
                f"./{label_names[y]}/{label_names[y]}_{cnts[y]}.wav",
                torch.transpose(chunk, 0, 1).numpy(),
                data[1].item(),
            )
            spec = spectrogram(chunk)
            plt.pcolor(spec[0].numpy())
            plt.savefig(f"./{label_names[y]}/{label_names[y]}_{cnts[y]}.jpg")
            plt.clf()
            plt.close()


if __name__ == "__main__":
    main()
