# 10. TIMIT 데이터셋 (TIMIT Acoustic-Phonetic Continuous Speech Corpus)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.196–201

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | TIMIT 데이터셋 (TIMIT Acoustic-Phonetic Continuous Speech Corpus) |
| **카테고리** | 음성 |
| **데이터셋 목적** | 음성/음소 인식(ASR / Phoneme Recognition) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★☆☆☆ (2/5) |
| **파일 형태** | wav 파일 (+ txt 전사, 음소/단어 라벨) |
| **데이터셋 링크** | https://catalog.ldc.upenn.edu/LDC93S1 |
| **원문 PDF 표기 링크** | https://catalog.ldc.upenn.edu/LDC93S1 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, LDC 공식 카탈로그 및 코드 내 deepai.org/timit.zip 200 OK) |
| **교과서 연계** | 인공지능과 미래사회 p.71 (음성 데이터 전처리) |

미국 8개 방언 지역(DR1~DR8) 630명의 화자가 각각 10개의 영어 문장을 읽은 16kHz 음소·단어·문장 단위 음성 인식 표준 코퍼스입니다.

---

## 2. 데이터셋 특징

- **규모**: 8개 방언 × 630명 화자 × 10문장 = 총 **6,300개** 발화(16kHz).
- **문장 3종 구분**: `SA`(공통 2문장), `SX`(음소 균형 5문장), `SI`(화자별 상이 3문장).
- **파일 계층**: `{TRAIN|TEST}/DR1~DR8/{화자ID}/{문장ID}.{WAV.wav|TXT|PHN|WRD}` 구조.
- **편집 참고**: p.196 하단에 `데이터셋 목적` 라벨이 2번 인쇄되어 있으며, 두 번째 항목에 V09의 '이진분류 / 도메인 일반화'가 잘못 복붙되어 있습니다.

---

## 3. 데이터셋 구성

| 폴더 / 파일 유형 | 설명 |
|---|---|
| `DR1` ~ `DR8` | 미국 8개 방언 권역 |
| `*.WAV.wav` | 16kHz 음성 파일 |
| `*.TXT` | 시작 샘플, 끝 샘플, 정답 문장 텍스트 (예: `0 63488 She had your dark suit...`) |
| `*.PHN` / `*.WRD` | 음소(Phoneme) 및 단어(Word) 정렬 라벨 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: 방언-화자 계층 디렉토리 구조 및 `.TXT` 전사 파일 예시.

---

## 5. 데이터셋 활용

1. **음성 데이터 전처리 해보기** (`인공지능과 미래사회 p.71`).
2. **폴더 별로 분류된 데이터에서 원하는 데이터 추출하기**: DR/화자 폴더 순회 및 `.WAV.wav`와 `.TXT` 매칭 추출.
3. **음성 인식 기술에 대해 알아보기**: 음소(Phoneme) 단위 인식과 단어(Word) 단위 인식의 차이 탐구.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: 원문 코드(p.199~201)는 `https://data.deepai.org/timit.zip`(200 OK 확인)을 내려받아 `TEST/DR*/화자/` 폴더를 순회하며 100개 발화의 정답 문장, Mel-Spectrogram PNG, WAV를 추출합니다.

```python
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
```
