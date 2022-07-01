# -*- coding: utf-8 -*-
# @Time     :2021/12/7 17:10
# @File     :zengjiashuju.py
# @Software :PyCharm
# @Project  :增加数据集（图片翻转）
# @Content  :

import os
import cv2
import canshu



#水平垂直翻转
def fanzhuan(file_list,file_num):
    for file_name in file_list: #逐个获取图片文件名
        file_path = path + "/" + label + "/" + file_name #获取当前文件名路径
        img1 = cv2.imread(file_path) #读取图片
        img2 = cv2.flip(img1, 0) #垂直翻转
        img3 = cv2.flip(img1, 1) #水平翻转
        cv2.imshow("yuantu", img1)
        cv2.imshow("chuizhi", img2)
        cv2.imshow("shuiping", img3)
        cv2.waitKey(10)
        new_path_chuizhi = path + "/" + label + "/" + label + "_" + str(file_num) + ".jpg"  #垂直新图片文件名和路径
        new_path_shuiping = path + "/" + label + "/" + label + "_" + str(file_num+1) + ".jpg"  #水平新图片文件名和路径
        cv2.imwrite(new_path_chuizhi, img2)
        cv2.imwrite(new_path_shuiping,img3)
        file_num += 2


if __name__ == "__main__":
    path = canshu.train_img_dir  # 文件夹路径
    label = input("文件夹名称：")
    file_list = os.listdir(path + "/" + label)  # 获取指定目录下的所有图片文件名
    file_num = len(file_list)  # 文件个数


    fanzhuan(file_list,file_num) #翻转
    cv2.destroyAllWindows()
