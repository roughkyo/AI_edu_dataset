# 01. ESC-10 데이터셋 (Dataset for Environmental Sound Classification (ESC-10))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.142–148

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | ESC-10 데이터셋 (Dataset for Environmental Sound Classification (ESC-10)) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://github.com/karolpiczak/ESC-50 |
| **원문 PDF 표기 링크** | https://github.com/karolpiczak/ESC-50 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소 및 master.zip 200 OK) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

총 50개 환경음 클래스로 구성된 ESC-50에서 입문자용으로 10개 클래스를 추출한 5초 길이 환경음(코 고는 소리, 개 짖는 소리, 빗소리 등) 분류 데이터셋입니다.

---

## 2. 데이터셋 특징

- **규모**: 10개 클래스 × 클래스당 40개 = 총 **400개** `.wav` 파일(각 약 5초, 44.1kHz).
- **원본(ESC-50)과의 차이**: ESC-50(동물·자연·사람·실내·실외 총 50종 2,000개) 중 `esc10 == True` 플래그가 붙은 10개 클래스만 필터링해 추출합니다.

---

## 3. 데이터셋 구성

| 번호 | 영문 클래스명 | 한글 클래스명 |
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
| ⑩ | `chainsaw` | 전기톱 소리 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 10개 한글 클래스 폴더별 `.wav` 파일 (각 40개).
- **시각화 데이터**: `librosa.feature.melspectrogram` + `power_to_db`로 변환한 2차원 Mel-Spectrogram 이미지.

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 시간-주파수 변환으로 2차원 스펙트로그램 이미지 생성 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 등 이미지 분류 CNN 모델로 스펙트로그램 분류 학습.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.145~148)는 GitHub에서 `ESC-50` ZIP을 내려받아 `esc50.csv`의 `esc10 == True` 항목만 골라 스펙트로그램 PNG와 WAV를 클래스 폴더에 저장합니다.

```python
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
```
