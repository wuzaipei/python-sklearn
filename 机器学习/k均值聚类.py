# coding:utf-8
import sklearn.datasets as datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# 1、创建数据
#无监督学习，算法不需要我们告诉它类别。它自动分出类别
x_tain,target=datasets.make_blobs(100,centers=10)
print(x_tain[:5,:])
# 2、建立模型对数据进行训练
kmeans = KMeans()   #n_clusetrs 这个是设置你要分为多少类
#训练
kmeans.fit(x_tain,target)  #这个是无监督学习没有预测训练值
y_t=kmeans.predict(x_tain)
centers = kmeans.cluster_centers_

#首先绘制初始的数据
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.scatter(x_tain[:,0],x_tain[:,1],c=target)  # c 是设置类别的属性
plt.title('原来数据')
plt.subplot(1,2,2)
plt.scatter(x_tain[:,0],x_tain[:,1],c=y_t)
plt.title('预测数据')
plt.figure()
plt.scatter(list(range(len(y_t))),y_t,c=y_t)
plt.show()
