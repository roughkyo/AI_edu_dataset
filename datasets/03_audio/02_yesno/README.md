# 02. Yes/No 데이터셋 (Yes/No Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.149–154

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Yes/No 데이터셋 (Yes/No Dataset) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 이진 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★☆☆☆ (2/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://www.openslr.org/1/ |
| **원문 PDF 표기 링크** | https://www.openslr.org/1/ |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, OpenSLR 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

한 명의 화자가 히브리어로 'Yes' 또는 'No' 단어를 연속 8회 발음한 60개의 녹음 파일을 개별 단어로 분할해 이진 음성 분류에 활용하는 초경량 데이터셋입니다.

---

## 2. 데이터셋 특징

- **규모**: 원본 **60개** `.wav` 파일(8kHz, 각 8단어 포함) → 전처리(VAD 무음 제거 + 8등분)를 통해 **480개** 개별 단어 음성 생성.
- **편집 참고**: p.150 `데이터셋 구성` 상단에 'ESC-10 데이터셋은 총 10개의 클래스'라는 직전 단원 잔재 오기가 있으나 실제 클래스는 `Yes`, `No` 2개입니다.

---

## 3. 데이터셋 구성

| 라벨 | 클래스 | 설명 |
|---|---|---|
| `1` | `Yes` | 긍정에 해당하는 히브리어 음성 |
| `0` | `No` | 부정에 해당하는 히브리어 음성 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `yes/`, `no/` 폴더에 분할된 `.wav` 파일(`yes_1.wav` 등).
- **시각화 데이터**: 8kHz 기반 스펙트로그램(`n_fft=256, win_length=200, hop_length=80`).

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 스펙트로그램 이미지 변환 (`인공지능과 미래사회 p.71`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **전체 학습을 통해 음성 분류하기**: 2클래스 신경망 분류 성능 비교.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.152~154)는 `torchaudio.datasets.YESNO`를 내려받아 `torchaudio.functional.vad`로 앞뒤 무음을 잘라낸 뒤 8등분하여 WAV와 스펙트로그램 JPG로 저장합니다.

```python
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
```
