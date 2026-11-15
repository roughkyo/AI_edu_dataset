# Ⅲ. 음성 데이터셋 (Audio Datasets)

환경음·도시소음·음성명령어·음악장르 분류(6종), 연속 음성 인식 및 합성(3종: LJSpeech, LibriSpeech, TIMIT), 도메인 일반화(Bird Audio Detection) 총 10개 데이터셋을 수록합니다. 모든 데이터셋에 `torchaudio` 또는 `librosa`를 활용한 2차원 스펙트로그램 이미지 변환 및 WAV 추출 예시 코드가 포함되어 있습니다.

| 번호 | 데이터셋 명 | 데이터셋 목적 | 난이도 | 흥미도 | 형태 | 공식 링크 | 예시 코드 |
|---|---|---|---|---|---|---|---|
| 01 | [ESC-10 데이터셋](./01_esc10/README.md) | 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | wav 파일 | [링크](https://github.com/karolpiczak/ESC-50) | [코드](./01_esc10/prepare_dataset.py) |
| 02 | [Yes/No 데이터셋](./02_yesno/README.md) | 이진 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | wav 파일 | [링크](https://www.openslr.org/1/) | [코드](./02_yesno/prepare_dataset.py) |
| 03 | [FSD(Free Spoken Digit) 데이터셋](./03_free_spoken_digit/README.md) | 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | wav 파일 | [링크](https://github.com/Jakobovski/free-spoken-digit-dataset) | [코드](./03_free_spoken_digit/prepare_dataset.py) |
| 04 | [URBAN SOUND 데이터셋](./04_urbansound8k/README.md) | 음성 분류(Audio Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | wav 파일 | [링크](https://urbansounddataset.weebly.com/urbansound8k.html) | [코드](./04_urbansound8k/prepare_dataset.py) |
| 05 | [Speech Command 데이터셋](./05_speech_commands/README.md) | 분류(Classification - 키워드 음성 인식) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★★☆☆ (3/5) | wav 파일 | [링크](https://arxiv.org/abs/1804.03209) | [코드](./05_speech_commands/prepare_dataset.py) |
| 06 | [GTZan 데이터셋](./06_gtzan/README.md) | 분류(Classification - 음악 장르 인식 MGR) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★★★☆ (4/5) | wav 파일 | [링크](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) | [코드](./06_gtzan/prepare_dataset.py) |
| 07 | [LJSpeech 데이터셋](./07_ljspeech/README.md) | 음성 인식(ASR) / 지도학습(Supervised Learning) | ★★★★☆ (4/5) | ★★☆☆☆ (2/5) | wav 파일 (+ 전사 텍스트) | [링크](https://keithito.com/LJ-Speech-Dataset/) | [코드](./07_ljspeech/prepare_dataset.py) |
| 08 | [Librispeech 데이터셋](./08_librispeech/README.md) | 음성 인식(ASR) / 지도학습(Supervised Learning) | ★★★★☆ (4/5) | ★★★☆☆ (3/5) | wav 파일 (원본 flac -> 코드로 wav 추출) | [링크](http://www.openslr.org/11/) | [코드](./08_librispeech/prepare_dataset.py) |
| 09 | [Bird Audio Detection challenge 데이터셋](./09_bird_audio_detection/README.md) | 이진 분류(새소리 유무) / 지도학습 / 도메인 일반화(Domain Generalization) | ★★★☆☆ (3/5) | ★★★☆☆ (3/5) | wav 파일 (+ csv 메타데이터) | [링크](https://dcase.community/challenge2018/task-bird-audio-detection) | [코드](./09_bird_audio_detection/prepare_dataset.py) |
| 10 | [TIMIT 데이터셋](./10_timit/README.md) | 음성/음소 인식(ASR / Phoneme Recognition) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★☆☆☆ (2/5) | wav 파일 (+ txt 전사, 음소/단어 라벨) | [링크](https://catalog.ldc.upenn.edu/LDC93S1) | [코드](./10_timit/prepare_dataset.py) |
