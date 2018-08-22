# coding:utf-8
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from sklearn.decomposition import PCA

# 1、数据读取
data1=pd.read_excel('C:\\Users\wuzaipei\Desktop\泰迪杯赛题\A题\部分数据\附件1\谐波数据\YD_10.xlsx')

#PCA是主成分降维的构造器
data2 = data1.iloc[::,1:51]
data3 = data2

# 2、S主成分降维思想
# 里面的参数 n_coponentes 这个主要是取出多少个主成分来进行描述，whiten 主要是标准方差相同的问题
pca = PCA(n_components= 20,whiten= True,svd_solver='randomized')
#
pca.fit(data3) #里面可以传入需要降维的数据矩阵
data4=  pca.fit_transform(data3) #降维过后的数据
gxl = pca.explained_variance_ratio_   # 输出累计贡献率
# data4 = DataFrame(data4)  #这个是把数据转化为dataframe类型
data5 = data4.reshape(-1)
data5 = DataFrame(data5).T
print(data5.shape,'\n',type(data5))
print(sum(gxl))

# 3、矩阵缩放，特征不变
from scipy.misc import imresize
n_1 = np.array(data2)
# n_1 = np.random.randint(0,10,[20,20])
da_ta = imresize(data2, (100,50))
print(da_ta.shape)
print(da_ta[50:60,40::])
