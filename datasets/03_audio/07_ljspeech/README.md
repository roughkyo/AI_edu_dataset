# 07. LJSpeech 데이터셋 (The LJ Speech Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.179–183

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | LJSpeech 데이터셋 (The LJ Speech Dataset) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 음성 인식(ASR) / 지도학습(Supervised Learning) |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 (+ 전사 텍스트) |
| **데이터셋 링크** | https://keithito.com/LJ-Speech-Dataset/ |
| **원문 PDF 표기 링크** | https://keithito.com/LJ-Speech-Dataset/ |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Keith Ito 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

단일 여성 화자가 7권의 비문학 도서 구절을 낭독한 1~10초 길이의 13,100개 영어 발화 음성과 대본 텍스트로 구성된 음성 인식(ASR)·음성 합성(TTS) 연구용 데이터셋입니다.

---

## 2. 데이터셋 특징

- **규모**: 단일 화자 약 **13,100개** 발화(길이 1~10초, 22.05kHz).
- **특징**: 정해진 유한 클래스 분류가 아니라 연속된 문장 대본(`transcript`, `normalized_transcript`)을 예측·비교하는 심화 음성인식(ASR)용 데이터셋입니다.

---

## 3. 데이터셋 구성

| 구성 요소 (`torchaudio.datasets.LJSPEECH` 반환값) | 설명 |
|---|---|
| `waveform` | 음성 신호 텐서 |
| `sample_rate` | 샘플링 레이트 (22,050 Hz) |
| `transcript` | 원본 대본 문장 |
| `normalized_transcript` | 숫자/약어 등이 정규화된 대본 문장 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `./LJSPEECH/` 내 번호별 `.wav` 파일과 문장 대본 CSV (`label.csv`, `speech.csv`).

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **음성 인식(ASR) 기술과 세부 기술에 대해 알아보기 (심화)**:
   - Q1-1: ASR(Automatic Speech Recognition) 기술 이해
   - Q1-2: AM(Acoustic Model) 데이터 처리 흐름
   - Q1-3: LM(Language Model)과 단어/문장 단위 처리 차이

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.182~183)는 `torchaudio.datasets.LJSPEECH`에서 처음 100개 발화를 추출해 WAV, 스펙트로그램 JPG, 대본 CSV로 저장합니다.

```python
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
```
