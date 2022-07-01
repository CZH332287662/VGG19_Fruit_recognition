# -*- coding: utf-8 -*-
# @Time     :2021/12/6 16:39
# @File     :canshu.py
# @Software :PyCharm
# @Project  :参数信息
# @Content  :可变参数信息


"""路径参数"""
train_img_dir = "./sample/train" #训练集路径
test_img_dir = "./sample/test" #测试集路径
model_save_dir = "./model" #模型保存路径


"""种类参数"""
labels = ['Pear','Banana','Persimmon','Mangosteen','Durian']#, 'Carambola','Apple', 'Mango']#,  'Pear', 'Pomegranate'] #种类标签
num = len(labels) #种类数


"""图片参数"""
image_width = 64 #图片统一宽度
image_height = 64 #图片统一高度


"""训练参数"""
train_ratio = 0.9 #训练集比例
train_batch_size = 128 #训练集批次大小
val_batch_size = 128 #验证集批次大小
epochs = 60 #训练次数
lr = 0.0001 #学习率  0.000005




# """路径参数"""
# train_img_dir = "./sample/train" #训练集路径
# test_img_dir = "./sample/test" #测试集路径
# model_save_dir = "./model" #模型保存路径
#
#
# """种类参数"""
# labels = ['Apple', 'Banana', 'Carambola', 'Kiwi', 'Mango']#, 'Orange',  'Pear','Persimmon', 'Plum', 'Pomegranate'] #种类标签
# num = len(labels) #种类数
#
#
# """图片参数"""
# image_width = 64 #图片统一宽度
# image_height = 64 #图片统一高度
#
#
# """训练参数"""
# train_ratio = 0.9 #训练集比例
# train_batch_size = 128 #训练集批次大小
# val_batch_size = 128 #验证集批次大小
# epochs = 250 #训练次数
# lr = 0.00005 #学习率  0.000002