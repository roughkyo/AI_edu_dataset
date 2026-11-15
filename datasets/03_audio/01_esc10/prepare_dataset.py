# [V01] ESC-10 데이터셋 다운로드, 스펙트로그램 및 WAV 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.145-148

import os
import urllib.request
import zipfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def main():
    # 1. ESC-50 다운로드 및 압축 해제
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    urllib.request.urlretrieve(url, "./esc50.zip")
    with zipfile.ZipFile("./esc50.zip", "r") as zf:
        zf.extractall("./")

    # 2. 메타데이터 CSV 읽기
    df = pd.read_csv("./ESC-50-master/meta/esc50.csv", encoding="UTF-8")
    createFolder("./train")

    # 3. ESC-10 클래스 매핑 딕셔너리
    dicts = {
        "chainsaw": [0, "전기톱소리"],
        "clock_tick": [1, "시계바늘소리"],
        "crackling_fire": [2, "모닥불소리"],
        "crying_baby": [3, "아기울음소리"],
        "dog": [4, "개짖는소리"],
        "helicopter": [5, "헬리콥터소리"],
        "rain": [6, "빗소리"],
        "rooster": [7, "닭소리"],
        "sea_waves": [8, "파도소리"],
        "sneezing": [9, "코고는소리"],
    }
    cnt = np.zeros(10, dtype=int)

    # 4. esc10 == True 인 400개 샘플에 대해 Mel-Spectrogram PNG 및 WAV 저장
    for _, row in df.iterrows():
        if bool(row["esc10"]):
            cat = row["category"]
            j, kor_name = dicts[cat]
            cnt[j] += 1
            train_path = f"./train/{j}/"
            createFolder(train_path)

            file_name = row["filename"]
            x, _ = librosa.load(f"./ESC-50-master/audio/{file_name}", sr=44100)

            y = librosa.feature.melspectrogram(y=x, sr=44100, n_fft=2205, hop_length=441)
            log_spectrogram = librosa.power_to_db(y)

            plt.figure(figsize=(10, 4))
            plt.imshow(log_spectrogram, origin="lower", interpolation=None, cmap="viridis", aspect=1.1)
            plt.xlabel("Time")
            plt.ylabel("Frequency")
            plt.title("Spectrogram (dB)")
            plt.savefig(f"{train_path}{kor_name}_{cnt[j]}.png")
            plt.clf()
            plt.close()

            sf.write(f"{train_path}{kor_name}_{cnt[j]}.wav", x, 44100)


if __name__ == "__main__":
    main()
