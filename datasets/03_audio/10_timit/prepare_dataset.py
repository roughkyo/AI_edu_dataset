# [V10] TIMIT 데이터셋 다운로드, 방언-화자 계층 파싱 및 스펙트로그램 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.199-201
# [Colab/Shell] !wget https://data.deepai.org/timit.zip

import os
import urllib.request
import zipfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf


def main():
    # 1. zip 다운로드 및 압축 해제 (200 OK 확인됨)
    url = "https://data.deepai.org/timit.zip"
    urllib.request.urlretrieve(url, "./timit.zip")
    with zipfile.ZipFile("./timit.zip", "r") as zf:
        zf.extractall("./Timit")

    # 2. 방언(DR1~DR8) 및 화자 폴더 탐색
    test_path = "./Timit/data/TEST"
    test_dr_list = sorted(os.listdir(test_path))
    test_sp_list = [sorted(os.listdir(f"{test_path}/{dr}")) for dr in test_dr_list]

    test_wav_list = []
    test_txt_list = []
    for dr_num, dr in enumerate(test_dr_list):
        tmp_wav, tmp_txt = [], []
        for sp in test_sp_list[dr_num]:
            files = os.listdir(f"{test_path}/{dr}/{sp}")
            tmp_wav.append(sorted([f for f in files if f.endswith(".wav")]))
            tmp_txt.append(sorted([f for f in files if f.endswith(".TXT")]))
        test_wav_list.append(tmp_wav)
        test_txt_list.append(tmp_txt)

    # 3. 최대 100개 파일의 문장 전사, Mel-Spectrogram PNG 및 WAV 추출
    os.makedirs("./spectrogram", exist_ok=True)
    os.makedirs("./waveform", exist_ok=True)

    waveform = []
    utt = []
    file_num = 0

    for dr_num, dr in enumerate(test_dr_list):
        if file_num >= 100:
            break
        for sp_num, sp in enumerate(test_sp_list[dr_num]):
            if file_num >= 100:
                break
            for wavfile in test_wav_list[dr_num][sp_num]:
                if file_num >= 100:
                    break
                prefix = wavfile.split(".")[0]
                txt_candidates = [f for f in test_txt_list[dr_num][sp_num] if f.startswith(prefix)]
                if not txt_candidates:
                    continue
                txtfile = txt_candidates[0]

                with open(f"{test_path}/{dr}/{sp}/{txtfile}", "r", encoding="utf-8") as f_txt:
                    line = f_txt.readline().strip()
                wavlen = len(line.split(" ")[1])
                utterance = line[wavlen + 3 :]
                utt.append(utterance)

                x, _ = librosa.load(f"{test_path}/{dr}/{sp}/{wavfile}", sr=16000)
                waveform.append(x)

                y = librosa.feature.melspectrogram(y=x, sr=16000, n_fft=800, hop_length=160)
                log_spectrogram = librosa.power_to_db(y)

                plt.figure(figsize=(10, 4))
                plt.imshow(log_spectrogram, origin="lower", interpolation=None, cmap="viridis", aspect=1.1)
                plt.xlabel("Time")
                plt.ylabel("Frequency")
                plt.title("Spectrogram (dB)")
                plt.savefig(f"./spectrogram/{file_num}_{dr}_{sp}_{prefix}.png")
                plt.clf()
                plt.close()

                sf.write(f"./waveform/{file_num}_{dr}_{sp}_{prefix}.wav", x, 16000)
                file_num += 1


if __name__ == "__main__":
    main()
