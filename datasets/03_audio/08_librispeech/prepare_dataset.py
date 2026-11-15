# [V08] LibriSpeech (test-clean) 스펙트로그램 PNG 및 WAV 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.186-188

import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio


def main():
    datapath = "./DATA"
    os.makedirs(datapath, exist_ok=True)

    # 1. LibriSpeech test-clean 다운로드 (약 5.4시간)
    test_c_dataset = torchaudio.datasets.LIBRISPEECH(
        root=datapath, url="test-clean", download=True
    )

    os.makedirs("./spectrogram", exist_ok=True)
    os.makedirs("./waveform", exist_ok=True)

    # 2. 각 샘플을 Mel-Spectrogram PNG 및 WAV로 저장 (예시로 상위 50개 제한 가능)
    for idx, (waveform, sr, utt, sp_id, chap_id, utt_id) in enumerate(test_c_dataset):
        waveform_np = waveform.squeeze(0).numpy()
        y = librosa.feature.melspectrogram(y=waveform_np, sr=sr, n_fft=800, hop_length=160)
        log_spectrogram = librosa.power_to_db(y)

        plt.figure(figsize=(10, 4))
        plt.imshow(log_spectrogram, origin="lower", interpolation=None, cmap="viridis", aspect=1.1)
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("Spectrogram (dB)")
        plt.savefig(f"./spectrogram/{sp_id}_{utt_id}.png")
        plt.clf()
        plt.close()

        waveform_tp = torch.transpose(waveform, 0, 1).numpy()
        sf.write(f"./waveform/{sp_id}_{utt_id}.wav", waveform_tp, sr)

        if idx >= 49:
            break


if __name__ == "__main__":
    main()
