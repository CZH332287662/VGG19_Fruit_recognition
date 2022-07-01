# -*- coding: utf-8 -*-
# @Time     :2021/12/4 18:19
# @File     :yuchuli.py
# @Software :PyCharm
# @Project  :数据加载和预处理
# @Content  :
"""通过read_path方法读取文件夹中的所有子文件夹的图片文件路径，对路径进行打乱，随后划分训练集和验证集，
通过preprocess_img_all方法进行图片的加载以及预处理（调整图片大小、归一化、灰度并增加维度），
通过text2vec方法将图片标签进行独热编码转换，随后将图片矩阵数据和独热编码标签打包并分批"""

import os
import numpy as np
import tensorflow as tf
import pathlib
import random
import canshu

#数据获取并预处理
class ImageDataSet:
    def __init__(self):
        self.num = canshu.num #种类数
        self.train_batch_size = canshu.train_batch_size #训练批次大小
        self.val_batch_size = canshu.val_batch_size #验证批次大小

        data_root = pathlib.Path(canshu.train_img_dir) #创建训练图片路径对象
        image_paths = list(data_root.glob("*")) #遍历匹配所有（*）目录并存放到列表中
        #print("path:{}".format(image_paths))
        image_paths = self.read_path(image_paths) #获取所有图片文件路径（含文件名）
        print(image_paths,type(image_paths))
        random.shuffle(image_paths) #随机打乱

        #划分训练集验证集
        self.train_image_paths = image_paths[:int(canshu.train_ratio * len(image_paths))] #训练集
        self.val_image_paths = image_paths[int(canshu.train_ratio * len(image_paths)):] #验证集
        print("训练集大小：{}\t验证集大小：{}".format(len(self.train_image_paths),len(self.val_image_paths)))


    #数据训练前处理并打包
    def build(self):
        """
        :return: 预处理后的训练集和验证集
        """
        #训练集
        img_train_list = self.preprocess_img_all(self.train_image_paths) #加载图片并预处理
        #print(img_train_list)
        train_image_ds = tf.data.Dataset.from_tensor_slices(img_train_list) #训练集数据打包
        train_all_images_labels = [self.text2vec(pathlib.Path(path).name.split("_")[0]) for path in self.train_image_paths] #获取独热编码
        #print(train_all_images_labels)
        train_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(train_all_images_labels, tf.int32)) #训练集标签打包
        train_set = tf.data.Dataset.zip((train_image_ds, train_label_ds)) #合并数据和标签
        train_set = train_set.batch(self.train_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE) #训练前处理操作

        #验证集
        img_val_list = self.preprocess_img_all(self.val_image_paths)
        val_image_ds = tf.data.Dataset.from_tensor_slices(img_val_list)
        val_all_images_labels = [self.text2vec(pathlib.Path(path).name.split("_")[0]) for path in self.val_image_paths]
        val_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(val_all_images_labels, tf.int32))
        val_set = tf.data.Dataset.zip((val_image_ds, val_label_ds))
        val_set = val_set.batch(self.val_batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return train_set, val_set


    #所有图片预处理
    def preprocess_img_all(self, paths):
        """
        :param paths: 照片路径（含文件名）
        :return: 灰度后的图片矩阵
        """
        img_list = [] #存储图片矩阵
        for path in paths: #逐个图片获取
            #print(path)
            image = tf.io.read_file(path)  #读取图片文件
            image = tf.image.decode_png(image, channels=3)  #解码图像所需的彩色通道数目,输出RGB图像
            image = tf.image.resize(image, [canshu.image_height,canshu.image_width])  #调整图片大小
            image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1  #归一化
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2] #获取各个通道像素值
            image = 0.2989 * r + 0.5870 * g + 0.1140 * b #灰度处理
            image = tf.expand_dims(image, axis=2)  #增加维度，灰度后维度降低
            img_list.append(image)
        return img_list


    #标签转为独热编码
    def text2vec(self,text):
        """
        :param text: 标签值
        :return: 独热编码
        """
        text_num = canshu.labels.index(text) #获取标签列表的下标索引作为标签的唯一数字标识
        vector = np.zeros(self.num)  #初始化独热编码矩阵
        vector[text_num] = 1  #对应位置置一
        #print(vector)
        return vector


    #获取所有图片文件路径（含文件名）
    def read_path(self,paths):
        """
        :param paths: 父文件夹路径
        :return: 所有子孙图片文件
        """
        file_all = [] #存储图片文件路径
        for i in range(len(paths)): #逐个路径获取
            for file_name in os.listdir(paths[i]):  #逐个获取指定目录下的所有子目录和文件名
                full_path = os.path.abspath(os.path.join(paths[i],file_name))#合并路径
                if os.path.isdir(full_path):  #判断路径对象是否为文件夹
                    #如果是文件夹，递归调用获取其子文件
                    files = read_path(full_path) #递归
                    file_all.extend(files) #追加子文件夹中的图片文件列表
                else: #路径对象为文件
                    if file_name.endswith('.jpg') or file_name.endswith('.bmp') or file_name.endswith('.png'): #文件后缀是否为'.jpg'、'.bmp'、'.png'（是否为图片文件）
                        file_all.append(full_path)
        #print(file_all)
        return file_all

if __name__ == "__main__":
    ImageDataSet().build()