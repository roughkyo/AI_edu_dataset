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
