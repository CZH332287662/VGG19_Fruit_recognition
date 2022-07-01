# VGG19_Fruit_recognition

基于VGG19的水果识别，水果种类：香蕉、榴莲、山竹、梨、柿子。验证集精度95.51%，测试集精度92.00%

### 文件说明：

|- model:存放模型

|_ model1.h5:保存的模型

|- sample：存放数据集

|_ test：测试集

|_ train：训练集（含验证集，在yuchuli.py中划分）

|- canshu.py：相关参数

|- gaiming.py：批量修改图片文件命名

|- model.py：模型定义

|- test.py：测试

|- xunlian.py：训练

|- yuchuli.py：数据加载与预处理

|_ zengjiashuju.py：增加数据集（图像翻转）

### 数据集：

数据集下载链接：https://pan.baidu.com/s/1aw_cmqhYXEBp3eprVtY-KA 
提取码：9e89

### 模型：

模型下载链接：https://pan.baidu.com/s/13Cu6VMv9cE_filsfjmWKsA 
提取码：fwi5

### 步骤：

![image-20220701141631928](C:\Users\64228\AppData\Roaming\Typora\typora-user-images\image-20220701141631928.png)

### 训练及测试结果：

![image-20220701141730589](C:\Users\64228\AppData\Roaming\Typora\typora-user-images\image-20220701141730589.png)

![image-20220701141736236](C:\Users\64228\AppData\Roaming\Typora\typora-user-images\image-20220701141736236.png)