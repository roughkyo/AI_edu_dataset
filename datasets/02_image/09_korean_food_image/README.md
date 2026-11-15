# 09. 한국 음식 이미지 데이터셋 (Korean Food Image Dataset)

> 출처: 한국교육학술정보원(KERIS), 『인공지능 데이터셋 활용집』(2021), p.126–132

## 1. 기본 정보

| 항목 | 내용 |
|---|---|
| **데이터셋 명** | 한국 음식 이미지 데이터셋 (Korean Food Image Dataset) |
| **카테고리** | 이미지 |
| **데이터셋 목적** | 이미지 분류(Classification) / 지도학습(Supervised Learning) |
| **난이도** | ★★★☆☆ (3/5) |
| **흥미도** | ★★★★☆ (4/5) |
| **파일 형태** | jpg 파일 |
| **데이터셋 링크** | https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=79 |
| **원문 PDF 표기 링크** | https://aihub.or.kr/aidata/13594 |
| **링크 검증 상태 (2026-11)** | 유효 (200 OK, AI Hub 공식 페이지로 자동 리다이렉트) |
| **교과서 연계** | 없음 |

한국과학기술원에서 구축한 한국 음식 150종(약 15만 장) 이미지 데이터셋으로, 본 자료집에서는 교육용으로 12개 음식 클래스를 선별·축소해 활용합니다.

---

## 2. 데이터셋 특징

- **원본 규모**: 27종 음식 대분류(구이, 김치, 떡, 면, 밥 등)에 속하는 총 **150종**, 종목별 약 1,000장 = 총 **150,000장**.
- **원본 데이터와의 차이**: 학생 실습 편의를 위해 12개 클래스만 선정하고, 각 클래스당 **학습 500장 + 테스트 100장 = 600장(총 7,200장)**, 최대 해상도 **256×256** 이하로 축소 가공하는 코드를 제공합니다.

---

## 3. 데이터셋 구성

| 번호 | 음식 클래스 (축소 12종) | 번호 | 음식 클래스 |
|---|---|---|---|
| ① | 삼겹살 | ⑦ | 육개장 |
| ② | 깍두기 | ⑧ | 가지볶음 |
| ③ | 꿀떡 | ⑨ | 만두 |
| ④ | 라면 | ⑩ | 도토리묵 |
| ⑤ | 주먹밥 | ⑪ | 떡볶이 |
| ⑥ | 보쌈 | ⑫ | 식혜 |

---

## 4. 데이터셋 예시

- **원 데이터(Raw Data)**: `train/`, `test/` 아래 12개 한글 음식명 폴더별로 `{음식명}_{번호}.jpg` 형태로 저장됩니다.

---

## 5. 데이터셋 활용

1. **온전한 학습을 통해 한국 음식 이미지 데이터셋 분류하기**.
2. **부분 학습을 통해 한국 음식 이미지 데이터셋 분류하기**.

---

## 6. 예시 코드

- 파일: [prepare_dataset.py](./prepare_dataset.py)
- 참고사항: AI Hub 회원가입 및 승인 후 `kfood.zip`을 내려받아야 실행할 수 있으며, 원문 코드(p.129~132)는 한글 파일 경로 처리를 위해 `np.fromfile` + `cv2.imdecode` / `cv2.imencode` 우회 함수를 포함하고 있습니다.

```python
# [I09] 한국 음식 이미지 데이터셋 압축 해제, 12종 추출 및 256px 리사이징 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.129-132
# 주의: AI Hub(https://aihub.or.kr/aidata/13594)에서 사전 신청 후 kfood.zip을 내려받아야 합니다.

import glob
import os
import zipfile
import cv2
import numpy as np


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
    try:
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None


def imwrite(filename, img, params=None):
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode="w+b") as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(e)
        return False


def main():
    category_list = ["구이", "김치", "떡", "면", "밥", "찜", "국", "나물", "만두", "무침", "볶음", "음청류"]
    food_list = ["삼겹살", "깍두기", "꿀떡", "라면", "주먹밥", "보쌈", "육개장", "가지볶음", "만두", "도토리묵", "떡볶이", "식혜"]

    zips = zipfile.ZipFile("./한국 음식 이미지/kfood.zip")
    directory = "./한국 음식 이미지/"
    createFolder("./한국 음식 이미지/images")

    zipInfo = zips.infolist()
    for member in zipInfo:
        member.filename = member.filename.encode("cp437").decode("euc-kr")
        category = member.filename.split(".")[0]
        if category in category_list:
            i = category_list.index(category)
            zips.extract(member, directory)

            cate_zips = zipfile.ZipFile(f"{directory}{category_list[i]}.zip")
            for fileName in cate_zips.namelist():
                if food_list[i] in fileName:
                    cate_zips.extract(fileName, f"./한국 음식 이미지/images/{food_list[i]}")

            cnt = 0
            images = glob.glob(f"./한국 음식 이미지/images/{category_list[i]}/{food_list[i]}/*.jpg")
            createFolder("./train")
            createFolder("./test")
            createFolder(f"./train/{food_list[i]}")
            createFolder(f"./test/{food_list[i]}")

            for fname in images:
                img = imread(fname)
                if img is None:
                    continue
                cnt += 1
                height, width, _ = img.shape
                if width > height:
                    height = int(height * 256 / width)
                    width = 256
                else:
                    width = int(width * 256 / height)
                    height = 256
                img = cv2.resize(img, (width, height))

                if cnt <= 500:
                    imwrite(f"./train/{food_list[i]}/{food_list[i]}_{cnt}.jpg", img)
                elif cnt <= 600:
                    imwrite(f"./test/{food_list[i]}/{food_list[i]}_{cnt - 500}.jpg", img)
                else:
                    break


if __name__ == "__main__":
    main()
```
