# [I02] Fashion MNIST 데이터셋 추출 및 폴더 저장 예제
# 출처: KERIS 『인공지능 데이터셋 활용집』(2021), p.085-086

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
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankel boot",
    ]

    # 1. Fashion MNIST 데이터셋 내려받기
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

    # 2. 저장 폴더 생성
    createFolder("./train")
    createFolder("./test")

    # 3. 각 클래스별 train 600장, test 100장 jpg 저장
    for i in range(10):
        train_path = f"./train/{names[i]}/"
        test_path = f"./test/{names[i]}/"
        createFolder(train_path)
        createFolder(test_path)

        train_images = x_train[np.where(y_train == i)[0]]
        test_images = x_test[np.where(y_test == i)[0]]

        for j, train_img in enumerate(train_images):
            if j < 600:
                cv.imwrite(f"./train/{names[i]}/{j}.jpg", train_img)

        for k, test_img in enumerate(test_images):
            if k < 100:
                cv.imwrite(f"./test/{names[i]}/{k}.jpg", test_img)


if __name__ == "__main__":
    main()
