# Ⅱ. 이미지 데이터셋 (Image Datasets)

기초 이미지 분류(MNIST, Fashion MNIST, CIFAR10, Dogs vs. Cats, SVHN), 비지도·고해상도 표현 학습(STL-10), 객체 검출(PASCAL VOC), 스타일 이전(Horse to Zebra), 국내 특화 데이터(한국 음식, 한국어 글자체 OCR) 총 10개 데이터셋을 수록합니다.

> **참고**: 본문 순서 기준으로 `02 Fashion MNIST(p.081)`와 `03 CIFAR10(p.087)`을 배열했습니다(원문 p.2 목차 표기와 순서 반대). 또한 원문 링크가 만료(404)된 `01 MNIST` 및 `08 Horse to Zebra`는 동일 데이터셋의 공식 대체 링크로 업데이트했습니다.

| 번호 | 데이터셋 명 | 데이터셋 목적 | 난이도 | 흥미도 | 형태 | 공식 링크 | 예시 코드 |
|---|---|---|---|---|---|---|---|
| 01 | [MNIST 데이터셋](./01_mnist/README.md) | 이미지 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | jpg 파일 | [링크](https://storage.googleapis.com/cvdf-datasets/mnist/) | [코드](./01_mnist/prepare_dataset.py) |
| 02 | [Fashion MNIST 데이터셋](./02_fashion_mnist/README.md) | 이미지 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | jpg 파일 | [링크](https://github.com/zalandoresearch/fashion-mnist) | [코드](./02_fashion_mnist/prepare_dataset.py) |
| 03 | [CIFAR10 데이터셋](./03_cifar10/README.md) | 이미지 분류(Classification) / 지도학습(Supervised Learning) | ★★☆☆☆ (2/5) | ★★☆☆☆ (2/5) | jpg 파일 | [링크](https://www.cs.toronto.edu/~kriz/cifar.html) | [코드](./03_cifar10/prepare_dataset.py) |
| 04 | [Dogs vs. Cats 데이터셋](./04_dogs_vs_cats/README.md) | 이미지 분류(이진 분류) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★★☆☆ (3/5) | jpg 파일 | [링크](https://www.kaggle.com/c/dogs-vs-cats/overview) | [코드](./04_dogs_vs_cats/prepare_dataset.py) |
| 05 | [SVHN 데이터셋](./05_svhn/README.md) | 이미지 분류(Classification) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★★★☆ (4/5) | png+mat 파일 | [링크](http://ufldl.stanford.edu/housenumbers/) | [코드](./05_svhn/prepare_dataset.py) |
| 06 | [STL-10 데이터셋](./06_stl10/README.md) | 이미지 분류(지도학습) 및 비지도 표현 학습(Unsupervised Learning) | ★★★★☆ (4/5) | ★★★☆☆ (3/5) | mat/bin 파일 (추가로 png 등 저장 가능) | [링크](https://cs.stanford.edu/~acoates/stl10/) | [코드](./06_stl10/prepare_dataset.py) |
| 07 | [PASCAL VOC 데이터셋](./07_pascal_voc/README.md) | 객체 검출(Object Detection) / 지도학습 | ★★★★★ (5/5) | ★★★★☆ (4/5) | jpg 파일 (+ xml 어노테이션) | [링크](https://pjreddie.com/projects/pascal-voc-dataset-mirror/) | [코드](./07_pascal_voc/prepare_dataset.py) |
| 08 | [Horse to Zebra 데이터셋](./08_horse_to_zebra/README.md) | 스타일 이전(Style Transfer) / 비지도학습(Unsupervised Learning) | ★★★★☆ (4/5) | ★★★☆☆ (3/5) | jpg+csv 파일 | [링크](http://efrosgans.eecs.berkeley.edu/cyclegan/datasets/) | [코드](./08_horse_to_zebra/prepare_dataset.py) |
| 09 | [한국 음식 이미지 데이터셋](./09_korean_food_image/README.md) | 이미지 분류(Classification) / 지도학습(Supervised Learning) | ★★★☆☆ (3/5) | ★★★★☆ (4/5) | jpg 파일 | [링크](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=79) | [코드](./09_korean_food_image/prepare_dataset.py) |
| 10 | [한국어 글자체 이미지 데이터셋](./10_korean_font_image/README.md) | 이미지 분류 / 한글 광학 문자 인식(OCR) / 지도학습 | ★★★☆☆ (3/5) | ★★★★☆ (4/5) | png 파일 (+ json 라벨 및 어노테이션) | [링크](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=81) | [코드](./10_korean_font_image/prepare_dataset.py) |
