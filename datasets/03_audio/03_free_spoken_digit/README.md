# 03. FSD(Free Spoken Digit) 데이터셋 (Free Spoken Digit Dataset (FSD))

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.155–160

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | FSD(Free Spoken Digit) 데이터셋 (Free Spoken Digit Dataset (FSD)) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://github.com/Jakobovski/free-spoken-digit-dataset |
| **원문 PDF 표기 링크** | https://github.com/Jakobovski/free-spoken-digit-dataset.git |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, GitHub 공식 저장소) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

MNIST의 음성 버전에 해당하는 데이터셋으로, 6명의 화자가 0부터 9까지의 영문 한 자리 숫자를 발음한 8kHz `.wav` 파일 3,000개로 구성됩니다.

---

## 2. 데이터셋 특징

- **규모**: 10클래스(0~9) × 화자 6명 × 숫자당 50개 = 총 **3,000개** `.wav` 파일(샘플링 레이트 8kHz, 짧은 길이로 용량 부담이 매우 적음).
- **파일명 규칙**: `{라벨숫자}_{화자이름}_{샘플번호}.wav` (예: `3_george_23.wav`). 별도 라벨 파일 없이 파일명을 파싱해 라벨을 추출합니다.

---

## 3. 데이터셋 구성

| 클래스 | 설명 |
|---|---|
| `0, 1, 2, 3, 4, 5, 6, 7, 8, 9` (10클래스) | 8kHz 영어 숫자 발음 녹음 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `recordings/` 폴더 내 `0_jackson_47.wav` 등 3,000개 파일.
- **시각화 데이터**: 짧은 시간 축을 가진 8kHz 기반 스펙트로그램(`n_fft=400, hop_length=80`).

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 스펙트로그램 PNG 변환 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 등 이미지 분류 모델을 이용한 숫자 음성 분류.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.158~159)는 GitHub 저장소 ZIP(`free-spoken-digit-dataset-master.zip`)을 압축 해제한 뒤 파일명에서 숫자 라벨을 추출해 `./spectrogram/{0..9}/` 및 `./waveform/{0..9}/`에 정리합니다.

```python
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
```
