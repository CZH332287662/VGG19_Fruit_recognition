# -*- coding: utf-8 -*-
# @Time     :2021/12/6 19:46
# @File     :test.py
# @Software :PyCharm
# @Project  :
# @Content  :


import os
import numpy as np
import pathlib
import canshu
import tensorflow as tf


#测试集数据获取并预处理
class TestDataSet:
    def __init__(self):
        self.num = canshu.num #种类数

        self.image_paths = self.read_path(canshu.test_img_dir) #获取所有图片文件路径（含文件名）
        print("测试集大小：{}".format(len(self.image_paths)))


    #数据测试打包并预处理
    def build(self):
        """
        :return: 预处理后的测试集
        """
        #测试集
        img_test_list = self.preprocess_img_all(self.image_paths) #加载图片并预处理
        #print(img_test_list)
        test_image_ds = tf.data.Dataset.from_tensor_slices(img_test_list) #测试集数据打包
        test_all_images_labels = [self.text2vec(pathlib.Path(path).name.split("_")[0]) for path in self.image_paths] #获取独热编码
        #print(test_all_images_labels)
        test_label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(test_all_images_labels, tf.int32)) #测试集标签打包
        test_set = tf.data.Dataset.zip((test_image_ds, test_label_ds)) #合并数据和标签
        test_set = test_set.batch(25, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        return test_set

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
        :param paths: 文件夹路径
        :return: 所有图片文件
        """
        file_all = [] #存储图片文件路径
        for file_name in os.listdir(paths): #逐个图片获取
            full_path = os.path.abspath(os.path.join(paths, file_name))  #合并路径
            if file_name.endswith('.jpg') or file_name.endswith('.bmp') or file_name.endswith('.png'): #文件后缀是否为'.jpg'、'.bmp'、'.png'（是否为图片文件）
                file_all.append(full_path)
        #print(file_all)
        return file_all

if __name__ == "__main__":
    test_data = TestDataSet().build()
    model = tf.keras.models.load_model("./model/model1.h5")
    out = model.evaluate(test_data)
    print(out)