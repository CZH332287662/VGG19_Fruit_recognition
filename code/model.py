# -*- coding: utf-8 -*-
# @Time     :2021/12/4 17:47
# @File     :model.py
# @Software :PyCharm
# @Project  :模型
# @Content  :模型函数

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#模型构建
def model_CNN(num):
    # model = keras.models.Sequential([
    #
    #     layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), activation='relu',padding="same"),
    #     layers.BatchNormalization(),
    #     layers.MaxPool2D(2,2),
    #
    #     layers.Conv2D(64, kernel_size=(3,3), strides=(1, 1), activation='relu', padding="same"),
    #     layers.BatchNormalization(),
    #     layers.MaxPool2D(2,2),
    #
    #     layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    #     layers.BatchNormalization(),
    #     layers.MaxPool2D(2,2),
    #
    #     layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    #     layers.BatchNormalization(),
    #     layers.MaxPool2D(2,2),
    #
    #     layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
    #     layers.BatchNormalization(),
    #     layers.MaxPool2D(2, 2),
    #
    #     layers.Flatten(),
    #     layers.Dense(128, activation='relu'),
    #     layers.Dropout(0.5),
    #
    #     layers.Dense(64, activation='relu'),
    #     layers.Dropout(0.5),
    #
    #     layers.Dense(num, activation='softmax')
    # ])
    # return model

    #VGG13
    # model = keras.Sequential([
    #     layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Flatten(),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(num, activation="softmax")
    # ])
    # return model


    # # VGG19
    # model = keras.Sequential([
    #     layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
    #     layers.MaxPool2D(2, 2),
    #     layers.Flatten(),
    #     layers.Dense(256, activation="relu"),#256
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(num, activation="softmax")
    # ])
    # return model

    # VGG19
    model = keras.Sequential([
        layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(64, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(128, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(256, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.Conv2D(512, 3, strides=(1, 1), padding="same", activation="relu"),
        layers.MaxPool2D(2, 2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),  # 256
        layers.Dense(128, activation="relu"),
        layers.Dense(num, activation="softmax")
    ])
    return model
