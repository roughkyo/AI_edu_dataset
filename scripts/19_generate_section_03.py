from pathlib import Path

ROOT = Path(r"c:\Users\ysj\Desktop\upload_all\AI_edu_dataset")
BASE = ROOT / "datasets" / "03_audio"
BASE.mkdir(parents=True, exist_ok=True)

ITEMS = [
    {
        "slug": "01_esc10",
        "num": "01",
        "title_kr": "ESC-10 데이터셋",
        "title_en": "Dataset for Environmental Sound Classification (ESC-10)",
        "pages": "p.142–148",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://github.com/karolpiczak/ESC-50",
        "url_pdf": "https://github.com/karolpiczak/ESC-50",
        "url_status": "유효 (200 OK, GitHub 공식 저장소 및 master.zip 200 OK)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "총 50개 환경음 클래스로 구성된 ESC-50에서 입문자용으로 10개 클래스를 추출한 5초 길이 환경음(코 고는 소리, 개 짖는 소리, 빗소리 등) 분류 데이터셋입니다.",
        "features": """- **규모**: 10개 클래스 × 클래스당 40개 = 총 **400개** `.wav` 파일(각 약 5초, 44.1kHz).
- **원본(ESC-50)과의 차이**: ESC-50(동물·자연·사람·실내·실외 총 50종 2,000개) 중 `esc10 == True` 플래그가 붙은 10개 클래스만 필터링해 추출합니다.""",
        "structure": """| 번호 | 영문 클래스명 | 한글 클래스명 |
|---|---|---|
| ① | `sneezing` | 코 고는 소리 (※ 원문 한글 라벨 표기 기준; 사전상 재채기/코골이 혼용) |
| ② | `dog` | 개 짖는 소리 |
| ③ | `clock_tick` | 시곗바늘 소리 |
| ④ | `crying_baby` | 아기 울음 소리 |
| ⑤ | `rooster` | 닭 울음 소리 |
| ⑥ | `rain` | 빗소리 |
| ⑦ | `sea_waves` | 파도 소리 |
| ⑧ | `crackling_fire` | 모닥불 소리 |
| ⑨ | `helicopter` | 헬리콥터 소리 |
| ⑩ | `chainsaw` | 전기톱 소리 |""",
        "example": """- **원 데이터(Raw Data)**: 10개 한글 클래스 폴더별 `.wav` 파일 (각 40개).
- **시각화 데이터**: `librosa.feature.melspectrogram` + `power_to_db`로 변환한 2차원 Mel-Spectrogram 이미지.""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 시간-주파수 변환으로 2차원 스펙트로그램 이미지 생성 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 등 이미지 분류 CNN 모델로 스펙트로그램 분류 학습.""",
        "code_note": "원문 코드(p.145~148)는 GitHub에서 `ESC-50` ZIP을 내려받아 `esc50.csv`의 `esc10 == True` 항목만 골라 스펙트로그램 PNG와 WAV를 클래스 폴더에 저장합니다.",
        "code": """# [V01] ESC-10 데이터셋 다운로드, 스펙트로그램 및 WAV 추출 예제
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
""",
    },
    {
        "slug": "02_yesno",
        "num": "02",
        "title_kr": "Yes/No 데이터셋",
        "title_en": "Yes/No Dataset",
        "pages": "p.149–154",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "이진 분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://www.openslr.org/1/",
        "url_pdf": "https://www.openslr.org/1/",
        "url_status": "유효 (200 OK, OpenSLR 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "한 명의 화자가 히브리어로 'Yes' 또는 'No' 단어를 연속 8회 발음한 60개의 녹음 파일을 개별 단어로 분할해 이진 음성 분류에 활용하는 초경량 데이터셋입니다.",
        "features": """- **규모**: 원본 **60개** `.wav` 파일(8kHz, 각 8단어 포함) → 전처리(VAD 무음 제거 + 8등분)를 통해 **480개** 개별 단어 음성 생성.
- **편집 참고**: p.150 `데이터셋 구성` 상단에 'ESC-10 데이터셋은 총 10개의 클래스'라는 직전 단원 잔재 오기가 있으나 실제 클래스는 `Yes`, `No` 2개입니다.""",
        "structure": """| 라벨 | 클래스 | 설명 |
|---|---|---|
| `1` | `Yes` | 긍정에 해당하는 히브리어 음성 |
| `0` | `No` | 부정에 해당하는 히브리어 음성 |""",
        "example": """- **원 데이터(Raw Data)**: `yes/`, `no/` 폴더에 분할된 `.wav` 파일(`yes_1.wav` 등).
- **시각화 데이터**: 8kHz 기반 스펙트로그램(`n_fft=256, win_length=200, hop_length=80`).""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 스펙트로그램 이미지 변환 (`인공지능과 미래사회 p.71`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **전체 학습을 통해 음성 분류하기**: 2클래스 신경망 분류 성능 비교.""",
        "code_note": "원문 코드(p.152~154)는 `torchaudio.datasets.YESNO`를 내려받아 `torchaudio.functional.vad`로 앞뒤 무음을 잘라낸 뒤 8등분하여 WAV와 스펙트로그램 JPG로 저장합니다.",
        "code": """# [V02] Yes/No 데이터셋 VAD 무음 제거 및 8등분 추출 예제
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
""",
    },
    {
        "slug": "03_free_spoken_digit",
        "num": "03",
        "title_kr": "FSD(Free Spoken Digit) 데이터셋",
        "title_en": "Free Spoken Digit Dataset (FSD)",
        "pages": "p.155–160",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "분류(Classification) / 지도학습(Supervised Learning)",
        "url": "https://github.com/Jakobovski/free-spoken-digit-dataset",
        "url_pdf": "https://github.com/Jakobovski/free-spoken-digit-dataset.git",
        "url_status": "유효 (200 OK, GitHub 공식 저장소)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "MNIST의 음성 버전에 해당하는 데이터셋으로, 6명의 화자가 0부터 9까지의 영문 한 자리 숫자를 발음한 8kHz `.wav` 파일 3,000개로 구성됩니다.",
        "features": """- **규모**: 10클래스(0~9) × 화자 6명 × 숫자당 50개 = 총 **3,000개** `.wav` 파일(샘플링 레이트 8kHz, 짧은 길이로 용량 부담이 매우 적음).
- **파일명 규칙**: `{라벨숫자}_{화자이름}_{샘플번호}.wav` (예: `3_george_23.wav`). 별도 라벨 파일 없이 파일명을 파싱해 라벨을 추출합니다.""",
        "structure": """| 클래스 | 설명 |
|---|---|
| `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` (10클래스) | 8kHz 영어 숫자 발음 녹음 |""",
        "example": """- **원 데이터(Raw Data)**: `recordings/` 폴더 내 `0_jackson_47.wav` 등 3,000개 파일.
- **시각화 데이터**: 짧은 시간 축을 가진 8kHz 기반 스펙트로그램(`n_fft=400, hop_length=80`).""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 스펙트로그램 PNG 변환 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 등 이미지 분류 모델을 이용한 숫자 음성 분류.""",
        "code_note": "원문 코드(p.158~159)는 GitHub 저장소 ZIP(`free-spoken-digit-dataset-master.zip`)을 압축 해제한 뒤 파일명에서 숫자 라벨을 추출해 `./spectrogram/{0..9}/` 및 `./waveform/{0..9}/`에 정리합니다.",
        "code": """# [V03] FSD (Free Spoken Digit) 데이터셋 클래스별 WAV/스펙트로그램 추출 예제
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
""",
    },
    {
        "slug": "04_urbansound8k",
        "num": "04",
        "title_kr": "URBAN SOUND 데이터셋",
        "title_en": "URBAN SOUND Dataset (UrbanSound8K)",
        "pages": "p.161–167",
        "difficulty": "★★☆☆☆ (2/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "음성 분류(Audio Classification) / 지도학습(Supervised Learning)",
        "url": "https://urbansounddataset.weebly.com/urbansound8k.html",
        "url_pdf": "https://urbansounddataset.weebly.com/urbansound8k.html",
        "url_status": "유효 (200 OK, NYU UrbanSound8K 공식 페이지)",
        "textbook": "없음",
        "summary": "도시 환경에서 발생하는 10가지 소음(에어컨, 경적, 개 짖는 소리, 드릴, 사이렌 등)을 담은 8,732개(약 5.6GB, 4초 이내) 오디오 분류 데이터셋입니다.",
        "features": """- **규모**: 10개 클래스, 10개 fold로 구성된 **8,732개** `.wav` 파일(44.1/48kHz, 총 5.6GB).
- **핵심 메타데이터**: `UrbanSound8K.csv`에 `slice_file_name`, `fsID`, `start`, `end`, `salience`(1: foreground, 2: background), `fold`(1~10), `classID`, `class`가 기록되어 있습니다.""",
        "structure": """| classID | 클래스명 | 설명 |
|---|---|---|
| 0~9 (10종) | `air_conditioner`, `car_horn`, `children_playing`, `dog_bark`, `drilling`, `engine_idling`, `gun_shot`, `jackhammer`, `siren`, `street_music` | 도시 소음 10종 (4초 이하) |""",
        "example": """- **원 데이터(Raw Data)**: 10개 fold 폴더(`fold1`~`fold10`)별 `.wav` 파일과 `UrbanSound8K.csv`.
- **시각화 데이터**: 160,000 샘플 길이로 제로패딩 통일 후 변환한 스펙트로그램 이미지.""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 160,000 샘플 고정 길이 패딩/트림 및 스테레오→모노 평균화.
2. **음성 데이터 시각화 해보기**: 스펙트로그램 이미지 추출.
3. **전체 학습을 통해 음성 분류하기**: fold 기반 train(`1~9`)/test(`10`) 분할 학습.""",
        "code_note": "원문 코드(p.164~167)는 Colab에서 `opendatasets` + Kaggle API(`chrisfilo/urbansound8k`)로 데이터를 내려받아 `UrbanSoundDataset` 클래스로 길이 통일(160,000) 후 WAV 및 JPG 스펙트로그램으로 저장합니다.",
        "code": """# [V04] UrbanSound8K 데이터셋 PyTorch 전처리, 길이 통일 및 스펙트로그램 추출 예제
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
""",
    },
    {
        "slug": "05_speech_commands",
        "num": "05",
        "title_kr": "Speech Command 데이터셋",
        "title_en": "Speech Commands Dataset",
        "pages": "p.168–173",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "분류(Classification - 키워드 음성 인식) / 지도학습(Supervised Learning)",
        "url": "https://arxiv.org/abs/1804.03209",
        "url_pdf": "https://arxiv.org/abs/1804.03209",
        "url_status": "유효 (200 OK, arXiv 공식 논문 페이지)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "'헤이 구글', '시리'처럼 짧은 호출 명령어·단어를 인식하기 위해 구축된 35개 영어 단어 음성 데이터셋(16kHz, v0.02 기준)입니다.",
        "features": """- **규모**: 총 **35개 키워드**, 클래스당 약 1,600~4,000여 개 `.wav` 파일(16kHz).
- **원본 데이터와의 차이**: 전체 데이터가 방대하므로 예제 코드에서는 클래스별 10개씩(또는 원하는 개수)만 추출해 저장합니다.""",
        "structure": """| 구분 | 35개 키워드 목록 |
|---|---|
| 단어 목록 | `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero` |""",
        "example": """- **원 데이터(Raw Data)**: 35개 영문 단어별 폴더와 `backward_0.wav` 등의 오디오 파일.
- **시각화 데이터**: 16kHz 기반 스펙트로그램(`n_fft=512, win_length=400, hop_length=160`).""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 스펙트로그램 추출 (`인공지능과 미래사회 p.71`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **전체 학습을 통해 음성 분류하기**: 2개 이상의 클래스 선택 후 분류 모델 학습.""",
        "code_note": "원문 코드(p.171~172)는 `torchaudio.datasets.SPEECHCOMMANDS`를 내려받아 딕셔너리(`dicts`)로 클래스별 10개 미만 샘플만 WAV 및 스펙트로그램 JPG로 저장합니다.",
        "code": """# [V05] Speech Commands 데이터셋 클래스별 샘플 추출 및 스펙트로그램 저장 예제
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
""",
    },
    {
        "slug": "06_gtzan",
        "num": "06",
        "title_kr": "GTZan 데이터셋",
        "title_en": "GTZan Music Genre Classification",
        "pages": "p.174–178",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★★☆ (4/5)",
        "format": "wav 파일",
        "category": "음성",
        "purpose": "분류(Classification - 음악 장르 인식 MGR) / 지도학습(Supervised Learning)",
        "url": "https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification",
        "url_pdf": "https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification",
        "url_status": "유효 (200 OK, Kaggle 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "블루스, 클래식, 컨트리, 디스코, 힙합, 재즈, 메탈, 팝, 레게, 락 등 10개 음악 장르(각 30초, 장르당 100곡 총 1,000곡)를 분류하는 대표적 MGR(Music Genre Recognition) 데이터셋입니다.",
        "features": """- **규모**: 10개 장르 × 각 100개 = 총 **1,000개** `.wav` 파일(22.05kHz, 각 약 30초).
- **원본 데이터와의 차이**: 용량이 커서 예제 코드에서는 장르별 5개씩만 추출해 실습합니다.
- **편집 참고**: p.175 `원 데이터` 설명 첫 문장에 'Speech Command 데이터셋의 경우...'라는 직전 단원 잔재 오기가 있습니다.""",
        "structure": """| 번호 | 장르명 (10클래스) | 번호 | 장르명 |
|---|---|---|---|
| ① | 블루스 (`blues`) | ⑥ | 재즈 (`jazz`) |
| ② | 클래식 (`classical`) | ⑦ | 메탈 (`metal`) |
| ③ | 컨트리 (`country`) | ⑧ | 팝 (`pop`) |
| ④ | 디스코 (`disco`) | ⑨ | 레게 (`reggae`) |
| ⑤ | 힙합 (`hiphop`) | ⑩ | 락 (`rock`) |""",
        "example": """- **원 데이터(Raw Data)**: 10개 장르 폴더별 `classical.00000.wav` 등 30초 음원.
- **시각화 데이터**: 22kHz 기반 스펙트로그램(`n_fft=550, win_length=550, hop_length=220`).""",
        "utilization": """1. **음성 데이터 전처리 해보기**: 2차원 스펙트로그램 변환 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 기반 10장르 분류기 학습.""",
        "code_note": "원문 코드(p.176~178)는 `torchaudio.datasets.GTZAN`을 이용해 장르별 5곡 미만씩 WAV 및 스펙트로그램 JPG를 저장합니다.",
        "code": """# [V06] GTZan 데이터셋 다운로드 및 장르별 샘플 추출 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.176-178

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
    # 1. torchaudio로 GTZAN 다운로드
    audios = torchaudio.datasets.GTZAN("./", download=True)
    data_loader = torch.utils.data.DataLoader(audios, batch_size=1, shuffle=True)

    # 2. 22kHz 기반 Spectrogram 변환기
    spectrogram = nn.Sequential(
        AT.Spectrogram(n_fft=550, win_length=550, hop_length=220),
        AT.AmplitudeToDB(),
    )

    # 3. 장르별 최대 5개씩 WAV 및 스펙트로그램 저장
    dicts = {}
    for data in data_loader:
        audio = data[0][0]
        y = data[2][0]

        if y in dicts:
            dicts[y] += 1
        else:
            dicts[y] = 0
            createFolder(f"./{y}")

        if dicts[y] < 5:
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
""",
    },
    {
        "slug": "07_ljspeech",
        "num": "07",
        "title_kr": "LJSpeech 데이터셋",
        "title_en": "The LJ Speech Dataset",
        "pages": "p.179–183",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일 (+ 전사 텍스트)",
        "category": "음성",
        "purpose": "음성 인식(ASR) / 지도학습(Supervised Learning)",
        "url": "https://keithito.com/LJ-Speech-Dataset/",
        "url_pdf": "https://keithito.com/LJ-Speech-Dataset/",
        "url_status": "유효 (200 OK, Keith Ito 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "단일 여성 화자가 7권의 비문학 도서 구절을 낭독한 1~10초 길이의 13,100개 영어 발화 음성과 대본 텍스트로 구성된 음성 인식(ASR)·음성 합성(TTS) 연구용 데이터셋입니다.",
        "features": """- **규모**: 단일 화자 약 **13,100개** 발화(길이 1~10초, 22.05kHz).
- **특징**: 정해진 유한 클래스 분류가 아니라 연속된 문장 대본(`transcript`, `normalized_transcript`)을 예측·비교하는 심화 음성인식(ASR)용 데이터셋입니다.""",
        "structure": """| 구성 요소 (`torchaudio.datasets.LJSPEECH` 반환값) | 설명 |
|---|---|
| `waveform` | 음성 신호 텐서 |
| `sample_rate` | 샘플링 레이트 (22,050 Hz) |
| `transcript` | 원본 대본 문장 |
| `normalized_transcript` | 숫자/약어 등이 정규화된 대본 문장 |""",
        "example": """- **원 데이터(Raw Data)**: `./LJSPEECH/` 내 번호별 `.wav` 파일과 문장 대본 CSV (`label.csv`, `speech.csv`).""",
        "utilization": """1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **음성 인식(ASR) 기술과 세부 기술에 대해 알아보기 (심화)**:
   - Q1-1: ASR(Automatic Speech Recognition) 기술 이해
   - Q1-2: AM(Acoustic Model) 데이터 처리 흐름
   - Q1-3: LM(Language Model)과 단어/문장 단위 처리 차이""",
        "code_note": "원문 코드(p.182~183)는 `torchaudio.datasets.LJSPEECH`에서 처음 100개 발화를 추출해 WAV, 스펙트로그램 JPG, 대본 CSV로 저장합니다.",
        "code": """# [V07] LJSpeech 데이터셋 100개 샘플 추출 및 대본 CSV 저장 예제
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
""",
    },
    {
        "slug": "08_librispeech",
        "num": "08",
        "title_kr": "Librispeech 데이터셋",
        "title_en": "LibriSpeech ASR Corpus",
        "pages": "p.184–189",
        "difficulty": "★★★★☆ (4/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "wav 파일 (원본 flac -> 코드로 wav 추출)",
        "category": "음성",
        "purpose": "음성 인식(ASR) / 지도학습(Supervised Learning)",
        "url": "http://www.openslr.org/11/",
        "url_pdf": "http://www.openslr.org/11/",
        "url_status": "유효 (200 OK, OpenSLR 공식 페이지)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "참여형 오디오북 프로젝트 LibriVox에서 추출한 16kHz 다중 화자 영어 연속 음성 인식(ASR) 표준 코퍼스입니다.",
        "features": """- **서브셋 구성**: `dev-clean`, `test-clean`(5.4시간, 화자 40명), `dev-other`, `test-other`, `train-clean-100`(100시간), `train-clean-360`(360시간) 등.
- **입문용 가공**: `train-clean-100`만 해도 5GB가 넘어, 예제 코드에서는 경량 서브셋인 `test-clean`(5.4시간)을 내려받아 처리합니다.""",
        "structure": """| 반환 항목 (`torchaudio.datasets.LIBRISPEECH`) | 설명 |
|---|---|
| `waveform` | 16kHz 음성 텐서 (`1 x frame`) |
| `sample_rate` | 샘플링 레이트 (16,000 Hz) |
| `utterance` | 발화 문장 정답 텍스트 |
| `speaker_id` | 화자 고유 ID |
| `chapter_id` | 오디오북 챕터 ID |
| `utterance_id` | 문장 고유 ID |""",
        "example": """- **시각화 데이터**: `librosa.feature.melspectrogram(sr=16000, n_fft=800, hop_length=160)` 기반 Mel-Spectrogram.""",
        "utilization": """1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **음성 인식(ASR) 기술과 세부 기술에 대해 알아보기 (심화)**: Q1-1(ASR), Q1-2(AM), Q1-3(LM).""",
        "code_note": "원문 코드(p.186~188)는 `torchaudio.datasets.LIBRISPEECH`의 `test-clean`을 내려받아 `spectrogram` 폴더와 `waveform` 폴더에 `{speaker_id}_{utterance_id}` 파일명으로 저장합니다.",
        "code": """# [V08] LibriSpeech (test-clean) 스펙트로그램 PNG 및 WAV 추출 예제
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
""",
    },
    {
        "slug": "09_bird_audio_detection",
        "num": "09",
        "title_kr": "Bird Audio Detection challenge 데이터셋",
        "title_en": "Bird Audio Detection challenge dataset",
        "pages": "p.190–195",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★★☆☆ (3/5)",
        "format": "wav 파일 (+ csv 메타데이터)",
        "category": "음성",
        "purpose": "이진 분류(새소리 유무) / 지도학습 / 도메인 일반화(Domain Generalization)",
        "url": "https://dcase.community/challenge2018/task-bird-audio-detection",
        "url_pdf": "http://dcase.community/challenge2018/task-bird-audio-detection",
        "url_status": "유효 (200 OK, DCASE 공식 페이지로 리다이렉트)",
        "textbook": "없음",
        "summary": "서로 다른 지역·장비·환경에서 수집된 6개 음원 세트를 이용해 새소리 존재 여부(`hasbird`)를 판별하고 새로운 녹음 환경으로 일반화(Domain Adaptation)하는 DCASE 챌린지 데이터셋입니다.",
        "features": """- **도메인 일반화 전략**: 동일 데이터셋 내부 분할이 아닌 서로 다른 환경의 3개 개발 세트(train)와 3개 평가 세트(validation/test)로 교차 검증을 권장합니다.
- **규모**:
  - Development (3종, 총 **35,690개**): `freefield1010`(7,690개, 5.8GB), `warblrb10k`(8,000개, 4.3GB), `BirdVox-DCASE-20k`(20,000개, 15.4GB).
  - Evaluation (3종, 총 **12,620개**): `warblrb10k eval`(2,000개, 1.3GB), `Chernobyl`(6,620개, 5.3GB), `PolandNFC`(4,000개, 2.3GB).
- **편집 참고**: p.190 제목에 `"데이터셋 데이터셋"` 중복 오타가 있습니다.""",
        "structure": """| CSV 열 이름 | 설명 |
|---|---|
| `itemid` | 음성 파일 일련번호 (파일명) |
| `datasetid` | 소속 데이터셋 종류 (`warblrb10k` 등) |
| `hasbird` | 새 존재 여부 이진 라벨 (`0`: 없음, `1`: 있음) |""",
        "example": """- **원 데이터(Raw Data)**: 6개 데이터셋 압축 ZIP 파일 및 메타 CSV.
- **시각화 데이터**: 특정 음원의 스펙트로그램 예시.""",
        "utilization": """1. **음성 데이터 전처리 해보기** 및 **시각화 해보기**.
2. **전체 학습을 통해 음성 분류 / 도메인 일반화 적용해보기**: train/validation 분할 및 타 도메인 교차 검증 비교(Q3-1, Q3-2).""",
        "code_note": "원문 코드(p.193~195)는 Colab 환경에서 `warblrb10k_public_wav.zip`과 `warblrb10k_public_metadata_2018.csv`를 업로드한 뒤 `BirdSoundDataset`으로 로드하여 `Presence`/`Absence` 폴더별 WAV 및 스펙트로그램 JPG를 저장합니다.",
        "code": """# [V09] Bird Audio Detection (warblrb10k) PyTorch 전처리 및 스펙트로그램 저장 예제
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
""",
    },
    {
        "slug": "10_timit",
        "num": "10",
        "title_kr": "TIMIT 데이터셋",
        "title_en": "TIMIT Acoustic-Phonetic Continuous Speech Corpus",
        "pages": "p.196–201",
        "difficulty": "★★★☆☆ (3/5)",
        "interest": "★★☆☆☆ (2/5)",
        "format": "wav 파일 (+ txt 전사, 음소/단어 라벨)",
        "category": "음성",
        "purpose": "음성/음소 인식(ASR / Phoneme Recognition) / 지도학습(Supervised Learning)",
        "url": "https://catalog.ldc.upenn.edu/LDC93S1",
        "url_pdf": "https://catalog.ldc.upenn.edu/LDC93S1",
        "url_status": "유효 (200 OK, LDC 공식 카탈로그 및 코드 내 deepai.org/timit.zip 200 OK)",
        "textbook": "인공지능과 미래사회 p.71 (음성 데이터 전처리)",
        "summary": "미국 8개 방언 지역(DR1~DR8) 630명의 화자가 각각 10개의 영어 문장을 읽은 16kHz 음소·단어·문장 단위 음성 인식 표준 코퍼스입니다.",
        "features": """- **규모**: 8개 방언 × 630명 화자 × 10문장 = 총 **6,300개** 발화(16kHz).
- **문장 3종 구분**: `SA`(공통 2문장), `SX`(음소 균형 5문장), `SI`(화자별 상이 3문장).
- **파일 계층**: `{TRAIN|TEST}/DR1~DR8/{화자ID}/{문장ID}.{WAV.wav|TXT|PHN|WRD}` 구조.
- **편집 참고**: p.196 하단에 `데이터셋 목적` 라벨이 2번 인쇄되어 있으며, 두 번째 항목에 V09의 '이진분류 / 도메인 일반화'가 잘못 복붙되어 있습니다.""",
        "structure": """| 폴더 / 파일 유형 | 설명 |
|---|---|
| `DR1` ~ `DR8` | 미국 8개 방언 권역 |
| `*.WAV.wav` | 16kHz 음성 파일 |
| `*.TXT` | 시작 샘플, 끝 샘플, 정답 문장 텍스트 (예: `0 63488 She had your dark suit...`) |
| `*.PHN` / `*.WRD` | 음소(Phoneme) 및 단어(Word) 정렬 라벨 |""",
        "example": """- **원 데이터(Raw Data)**: 방언-화자 계층 디렉토리 구조 및 `.TXT` 전사 파일 예시.""",
        "utilization": """1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **폴더 별로 분류된 데이터에서 원하는 데이터 추출하기**: DR/화자 폴더 순회 및 `.WAV.wav`와 `.TXT` 매칭 추출.
3. **음성 인식 기술에 대해 알아보기**: 음소(Phoneme) 단위 인식과 단어(Word) 단위 인식의 차이 탐구.""",
        "code_note": "원문 코드(p.199~201)는 `https://data.deepai.org/timit.zip`(200 OK 확인)을 내려받아 `TEST/DR*/화자/` 폴더를 순회하며 100개 발화의 정답 문장, Mel-Spectrogram PNG, WAV를 추출합니다.",
        "code": """# [V10] TIMIT 데이터셋 다운로드, 방언-화자 계층 파싱 및 스펙트로그램 추출 예제
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
""",
    },
]


def render_readme(it: dict) -> str:
    return f"""# {it['num']}. {it['title_kr']} ({it['title_en']})

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), {it['pages']}

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | {it['title_kr']} ({it['title_en']}) |
| **카테고리** | {it['category']} |
| **데이터셋 목적** | {it['purpose']} |
| **난이도** | {it['difficulty']} |
| **흥미도** | {it['interest']} |
| **파일 형태** | {it['format']} |
| **데이터셋 링크** | {it['url']} |
| **원문 PDF 표기 링크** | {it['url_pdf']} |
| **링크 검증 상태 (2026-11)** | {it['url_status']} |
| **교과서 연계** | {it['textbook']} |

{it['summary']}

---

## 2. 데이터셋 특징

{it['features']}

---

## 3. 데이터셋 구성

{it['structure']}

---

## 4. 데이터셋 예시

{it['example']}

---

## 5. 데이터셋 활용

{it['utilization']}

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: {it['code_note']}

```python
{it['code'].strip()}
```
"""


def main() -> None:
    index_rows = []
    for it in ITEMS:
        d = BASE / it["slug"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "README.md").write_text(render_readme(it), encoding="utf-8")
        (d / "prepare_dataset.py").write_text(it["code"], encoding="utf-8")
        index_rows.append(
            f"| {it['num']} | [{it['title_kr']}](./{it['slug']}/README.md) | {it['purpose']} | {it['difficulty']} | {it['interest']} | {it['format']} | [링크]({it['url']}) | [코드](./{it['slug']}/prepare_dataset.py) |"
        )

    section_readme = f"""# Ⅲ. 음성 데이터셋 (Audio Datasets)

환경음·도시소음·음성명령어·음악장르 분류(6종), 연속 음성 인식 및 합성(3종: LJSpeech, LibriSpeech, TIMIT), 도메인 일반화(Bird Audio Detection) 총 10개 데이터셋을 수록합니다. 모든 데이터셋에 `torchaudio` 또는 `librosa`를 활용한 2차원 스펙트로그램 이미지 변환 및 WAV 추출 예시 코드가 포함되어 있습니다.

| 번호 | 데이터셋 명 | 데이터셋 목적 | 난이도 | 흥미도 | 형태 | 공식 링크 | 예시 코드 |
|---|---|---|---|---|---|---|---|
{chr(10).join(index_rows)}
"""
    (BASE / "README.md").write_text(section_readme, encoding="utf-8")
    print(f"Generated {len(ITEMS)} datasets in {BASE}")


if __name__ == "__main__":
    main()
