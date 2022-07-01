# -*- coding: utf-8 -*-
# @Time     :2021/12/4 21:46
# @File     :gaiming.py
# @Software :PyCharm
# @Project  :修改图片文件名称
# @Content  :

import os
import canshu


path = canshu.train_img_dir #文件路径
labels = canshu.labels #种类名称



for label in labels:
    j=0
    file_list = os.listdir(path + "/"+ label)  # 获取指定目录下的所有图片文件名
    #print(file_list)m
    for file_name in file_list: #逐个获取图片文件名
        old_path = path + "/" + label + "/" + file_name #获取当前文件名路径
        #print(old_path)
        new_path = path + "/" + label + "/" + label + "_" +str(j) + ".jpg" #新文件名及路径
        print(new_path)
        j += 1
        os.rename(old_path,new_path)  #修改文件名

