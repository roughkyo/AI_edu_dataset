# 06. GTZan 데이터셋 (GTZan Music Genre Classification)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.174–178

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | GTZan 데이터셋 (GTZan Music Genre Classification) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 분류(Classification - 음악 장르 인식 MGR) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification |
| **원문 PDF 표기 링크** | https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, Kaggle 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

블루스, 클래식, 컨트리, 디스코, 힙합, 재즈, 메탈, 팝, 레게, 락 등 10개 음악 장르(각 30초, 장르당 100곡 총 1,000곡)를 분류하는 대표적 MGR(Music Genre Recognition) 데이터셋입니다.

---

## 2. 데이터셋 특징

- **규모**: 10개 장르 × 각 100개 = 총 **1,000개** `.wav` 파일(22.05kHz, 각 약 30초).
- **원본 데이터와의 차이**: 용량이 커서 예제 코드에서는 장르별 5개씩만 추출해 실습합니다.
- **편집 참고**: p.175 `원 데이터` 설명 첫 문장에 'Speech Command 데이터셋의 경우...'라는 직전 단원 잔재 오기가 있습니다.

---

## 3. 데이터셋 구성

| 번호 | 장르명 (10클래스) | 번호 | 장르명 |
|---|---|---|---|
| ① | 블루스 (`blues`) | ⑥ | 재즈 (`jazz`) |
| ② | 클래식 (`classical`) | ⑦ | 메탈 (`metal`) |
| ③ | 컨트리 (`country`) | ⑧ | 팝 (`pop`) |
| ④ | 디스코 (`disco`) | ⑨ | 레게 (`reggae`) |
| ⑤ | 힙합 (`hiphop`) | ⑩ | 락 (`rock`) |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 10개 장르 폴더별 `classical.00000.wav` 등 30초 음원.
- **시각화 데이터**: 22kHz 기반 스펙트로그램(`n_fft=550, win_length=550, hop_length=220`).

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 2차원 스펙트로그램 변환 (`인공지능과 미래사회 p.71`).
2. **전체 학습을 통해 음성 분류하기**: VGG16 기반 10장르 분류기 학습.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.176~178)는 `torchaudio.datasets.GTZAN`을 이용해 장르별 5곡 미만씩 WAV 및 스펙트로그램 JPG를 저장합니다.

```python
# [V06] GTZan 데이터셋 다운로드 및 장르별 샘플 추출 예제
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
```
