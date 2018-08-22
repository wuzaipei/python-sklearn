import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
# import xlrd
#import xlrd 导入读取Excel的库

# 1、下面我们来读取原始数据使用pandas库中的read进行读取
# data=pd.read_excel('data1.xlsx',header=None)
data=pd.read_table('datingTestSet2.txt',header=None)
print(type(data),'\n',data.head())

# 2、从所有的数据中提取几列来做我们的训练数据在选取几列来做我们的预测数据。
biaoq=data.columns  #获取数据类型的列标价 index这个是获取行标签。
data1=data[biaoq[:3]]
data2=data[biaoq[3]]

# 3、切割数据
x_train=data1[:900]       #训练数据
x_predict=data1[900:]     #预测训练数据
y_train=data2[:900]       #预测训练数据
y_predict=data2[900:]     #实际结果

# print(x_train.head(),y_train.head())
# map方法可以对数据进行替换

# 4、下面是建立机器学习算法模型
knn = KNeighborsClassifier(6)  #默认参数是5，可以通过调整参数来提高预测准确度。
# 进行机械学习训练  y_train :这个里面的数据可以是string类型，这个参数不参与距离的计算
knn.fit(x_train,y_train)

y_pred=knn.predict(x_predict)  #预测的结果数据
# print(y_pred)
# print(y_predict.T)
gl=knn.score(x_predict,y_predict)   #第一个参数是预测的训练数据，第二个是实际结果数据
print('机器识别率为：',gl)

# 5、模型的保存
from sklearn.externals import joblib
joblib.dump(knn,'yc_mode.m')

# 6、导入模型
knn_1 = joblib.load('yc_mode.m')
y=knn_1.score(x_predict,y_predict)
print('机器识别度为：',y)

