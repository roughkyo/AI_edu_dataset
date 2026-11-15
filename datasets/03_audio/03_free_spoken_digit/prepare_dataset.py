# [V03] FSD (Free Spoken Digit) 데이터셋 클래스별 WAV/스펙트로그램 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.158-159
# 준비: https://github.com/Jakobovski/free-spoken-digit-dataset 에서 ZIP 다운로드

import os
import zipfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def main():
    # 1. ZIP 압축 해제
    with zipfile.ZipFile("./free-spoken-digit-dataset-master.zip", "r") as zf:
        zf.extractall("./")

    path_dir = "./free-spoken-digit-dataset-master/recordings"
    file_list = os.listdir(path_dir)

    # 2. 클래스별(0~9) 저장 폴더 생성
    for digit in range(10):
        os.makedirs(f"./spectrogram/{digit}", exist_ok=True)
        os.makedirs(f"./waveform/{digit}", exist_ok=True)

    # 3. 파일명 파싱 후 스펙트로그램 PNG 및 WAV 저장
    waveform = []
    digit_label = []

    for fname in file_list:
        if not fname.endswith(".wav"):
            continue
        digit = int(fname.split("_")[0])
        digit_label.append(digit)

        x, _ = librosa.load(f"{path_dir}/{fname}", sr=8000)
        waveform.append(x)

        y = librosa.feature.melspectrogram(y=x, sr=8000, n_fft=400, hop_length=80)
        log_spectrogram = librosa.power_to_db(y)

        plt.figure(figsize=(4, 4))
        plt.imshow(log_spectrogram, origin="lower", interpolation=None, cmap="viridis", aspect=1.1)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Spectrogram (dB)")
        stem = fname.split(".")[0]
        plt.savefig(f"./spectrogram/{digit}/{stem}.png")
        plt.clf()
        plt.close()

        sf.write(f"./waveform/{digit}/{fname}", x, 8000)


if __name__ == "__main__":
    main()
