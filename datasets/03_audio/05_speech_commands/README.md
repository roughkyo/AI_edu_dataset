# 05. Speech Command 데이터셋 (Speech Commands Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.168–173

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Speech Command 데이터셋 (Speech Commands Dataset) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 분류(Classification - 키워드 음성 인식) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | wav 파일 |
| **데이터셋 링크** | https://arxiv.org/abs/1804.03209 |
| **원문 PDF 표기 링크** | https://arxiv.org/abs/1804.03209 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, arXiv 공식 논문 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

'헤이 구글', '시리'처럼 짧은 호출 명령어·단어를 인식하기 위해 구축된 35개 영어 단어 음성 데이터셋(16kHz, v0.02 기준)입니다.

---

## 2. 데이터셋 특징

- **규모**: 총 **35개 키워드**, 클래스당 약 1,600~4,000여 개 `.wav` 파일(16kHz).
- **원본 데이터와의 차이**: 전체 데이터가 방대하므로 예제 코드에서는 클래스별 10개씩(또는 원하는 개수)만 추출해 저장합니다.

---

## 3. 데이터셋 구성

| 구분 | 35개 키워드 목록 |
|---|---|
| 단어 목록 | `backward`, `bed`, `bird`, `cat`, `dog`, `down`, `eight`, `five`, `follow`, `forward`, `four`, `go`, `happy`, `house`, `learn`, `left`, `marvin`, `nine`, `no`, `off`, `on`, `one`, `right`, `seven`, `sheila`, `six`, `stop`, `three`, `tree`, `two`, `up`, `visual`, `wow`, `yes`, `zero` |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 35개 영문 단어별 폴더와 `backward_0.wav` 등의 오디오 파일.
- **시각화 데이터**: 16kHz 기반 스펙트로그램(`n_fft=512, win_length=400, hop_length=160`).

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기**: 스펙트로그램 추출 (`인공지능과 미래사회 p.71`).
2. **학습/테스트 데이터 분류**: 7:3 또는 8:2 분할.
3. **전체 학습을 통해 음성 분류하기**: 2개 이상의 클래스 선택 후 분류 모델 학습.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.171~172)는 `torchaudio.datasets.SPEECHCOMMANDS`를 내려받아 딕셔너리(`dicts`)로 클래스별 10개 미만 샘플만 WAV 및 스펙트로그램 JPG로 저장합니다.

```python
# [V05] Speech Commands 데이터셋 클래스별 샘플 추출 및 스펙트로그램 저장 예제
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
```
