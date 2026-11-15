# [I03] CIFAR10 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.090-091

import os
import cv2 as cv
import numpy as np
import tensorflow as tf


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory. " + directory)


def main():
    names = [
        "Airplane", "Automobile", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck",
    ]

    # 1. CIFAR10 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스별 train 500장, test 100장 jpg 저장 (y_train은 (N, 1) 형태이므로 [0] 인덱싱)
    for i in range(10):
        train_path = f"./train/{names[i]}/"
        test_path = f"./test/{names[i]}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)[0]]
        test_images = x_test[np.where(y_test == i)[0]]

        for j, train_img in enumerate(train_images):
            if j < 500:
                # 참고: OpenCV는 BGR 순서를 사용하므로 색상을 보존하려면 cv.cvtColor(img, cv.COLOR_RGB2BGR) 권장
                cv.imwrite(f"./train/{names[i]}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{names[i]}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
