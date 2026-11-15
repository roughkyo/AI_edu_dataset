# 08. Librispeech 데이터셋 (LibriSpeech ASR Corpus)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.184–189

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | Librispeech 데이터셋 (LibriSpeech ASR Corpus) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 음성 인식(ASR) / 지도학습(Supervised Learning) |
| **난이도** | ★★★★☆ (4/5) |
| **흥미도** | ★★★☆☆ (3/5) |
| **파일 형태** | wav 파일 (원본 flac -> 코드로 wav 추출) |
| **데이터셋 링크** | http://www.openslr.org/11/ |
| **원문 PDF 표기 링크** | http://www.openslr.org/11/ |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, OpenSLR 공식 페이지) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

참여형 오디오북 프로젝트 LibriVox에서 추출한 16kHz 다중 화자 영어 연속 음성 인식(ASR) 표준 코퍼스입니다.

---

## 2. 데이터셋 특징

- **서브셋 구성**: `dev-clean`, `test-clean`(5.4시간, 화자 40명), `dev-other`, `test-other`, `train-clean-100`(100시간), `train-clean-360`(360시간) 등.
- **입문용 가공**: `train-clean-100`만 해도 5GB가 넘어, 예제 코드에서는 경량 서브셋인 `test-clean`(5.4시간)을 내려받아 처리합니다.

---

## 3. 데이터셋 구성

| 반환 항목 (`torchaudio.datasets.LIBRISPEECH`) | 설명 |
|---|---|
| `waveform` | 16kHz 음성 텐서 (`1 x frame`) |
| `sample_rate` | 샘플링 레이트 (16,000 Hz) |
| `utterance` | 발화 문장 정답 텍스트 |
| `speaker_id` | 화자 고유 ID |
| `chapter_id` | 오디오북 챕터 ID |
| `utterance_id` | 문장 고유 ID |

---

## 4. 데이터셋 예시

- **시각화 데이터**: `librosa.feature.melspectrogram(sr=16000, n_fft=800, hop_length=160)` 기반 Mel-Spectrogram.

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **음성 인식(ASR) 기술과 세부 기술에 대해 알아보기 (심화)**: Q1-1(ASR), Q1-2(AM), Q1-3(LM).

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.186~188)는 `torchaudio.datasets.LIBRISPEECH`의 `test-clean`을 내려받아 `spectrogram` 폴더와 `waveform` 폴더에 `{speaker_id}_{utterance_id}` 파일명으로 저장합니다.

```python
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
```
