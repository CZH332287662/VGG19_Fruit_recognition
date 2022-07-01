# VGG19_Fruit_recognition
基于VGG19的水果识别，水果种类：香蕉、榴莲、山竹、梨、柿子。验证集精度95.51%，测试集精度92.00%

文件说明：
|-model:存放模型
   |_model1.h5:保存的模型
|-sample：存放数据集
   |-test：测试集
   |_train：训练集（含验证集，在yuchuli.py中划分）
|-canshu.py：相关参数
|-gaiming.py：批量修改图片文件命名
|-model.py：模型定义
|-test.py：测试
|-xunlian.py：训练
|-yuchuli.py：数据加载与预处理
|_zengjiashuju.py：增加数据集（图像翻转）
