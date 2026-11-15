# [I01] MNIST 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.079-080
# 참고: 원본 Yann LeCun 서버의 .gz 파일이 404이므로 tf.keras 내장 로더 또는
#       공식 CVDF 미러(https://storage.googleapis.com/cvdf-datasets/mnist/)를 활용하세요.

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
    # 1. MNIST 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스(0~9)별로 train 500장, test 100장씩 jpg 저장
    for i in range(10):
        train_path = f"./train/{i}/"
        test_path = f"./test/{i}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)]
        test_images = x_test[np.where(y_test == i)]

        for j, train_img in enumerate(train_images):
            if j < 500:
                cv.imwrite(f"./train/{i}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{i}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
