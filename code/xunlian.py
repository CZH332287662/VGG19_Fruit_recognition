# -*- coding: utf-8 -*-
# @Time     :2021/12/4 22:11
# @File     :xunlian.py
# @Software :PyCharm
# @Project  :训练
# @Content  :对模型进行训练、保存以及训练期间的数据进行可视化展示



import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import optimizers,losses
from model import model_CNN
from yuchuli import ImageDataSet
import canshu
import matplotlib
matplotlib.rcParams["font.family"]="SimHei"
matplotlib.rcParams["font.sans-serif"] = "SimHei"


#训练
def Train():
    train_data,val_data = ImageDataSet().build() #数据获取并预处理

    #初始化模型
    model_save_dir = canshu.model_save_dir  #训练后的模型保存地址
    model = model_CNN(canshu.num)  #模型初始化

    #模型装配
    model.compile(
        optimizer=optimizers.Adam(lr=canshu.lr), #自适应估计随机梯度下降法
        loss=losses.categorical_crossentropy, #y值为独热编码形式的交叉熵损失函数
        metrics=['accuracy'] #以精确度作为衡量指标
    )
    history = model.fit(train_data,epochs=canshu.epochs,validation_data=val_data) #喂数据
    model.save(model_save_dir+"/model1.h5") #模型保存

    #绘制曲线
    plt.figure(figsize=(10,8))
    plt.subplot(221)
    plt.plot(history.history["loss"],color="r")
    plt.xlabel("训练次数（训练集）")
    plt.ylabel("损失")
    plt.title("训练次数-损失曲线（训练集）")
    plt.subplot(222)
    plt.plot(history.history["accuracy"],color="r")
    plt.ylim(0,1)
    plt.xlabel("训练次数（训练集）")
    plt.ylabel("精确度")
    plt.title("训练次数-精确度曲线（训练集）")
    plt.subplot(223)
    plt.plot(history.history["val_loss"], color="r")
    plt.xlabel("训练次数（验证集）")
    plt.ylabel("损失")
    plt.title("训练次数-损失曲线（验证集）")
    plt.subplot(224)
    plt.plot(history.history["val_accuracy"], color="r")
    plt.ylim(0,1)
    plt.xlabel("训练次数（验证集）")
    plt.ylabel("精确度")
    plt.title("训练次数-精确度曲线（验证集）")
    plt.show()


if __name__ == '__main__':
    #使用gpu训练
    gpu_ok = tf.test.is_gpu_available()
    print("use GPU", gpu_ok)

    # 训练
    Train()