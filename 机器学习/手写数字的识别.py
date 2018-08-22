# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1、数据读取
x_tain =[]
x_test =[]
y_tain=[]
y_test=[]
for i in range(0,10):
    for j in range(1,501):
        if j < 451: #将数据保存到训练数据中
            x_tain.append(plt.imread('./data/%d/%d_%d.bmp'%(i,i,j)).reshape(-1) )  #reshape 可以降维也就是矩阵变化
            y_tain.append(i)  #append 是读进来的数据进行存储的意思
        else: #保存到预测数据中
            x_test.append(plt.imread('./data/%d/%d_%d.bmp'%(i,i,j)).reshape(-1))
            y_test.append(i)

# 2、数据转换成
x_tain,y_tain= np.array(x_tain),np.array(y_tain)
# print(x_tain.shape,len(y_tain),len(x_test))

# 3、机器学习
knn = KNeighborsClassifier() #构造分类器
knn.fit(x_tain,y_tain)
y_ = knn.predict(x_test)  #进行预测的结果

# print(len(y_[::10]),'\n',y_test[::10])

gl=knn.score(x_test,y_test)
print('准确率为：',gl)

# 3、图片绘制
plt.figure(figsize=(13,15))
img = x_test[::10]
img1 = y_test[::10]
yimg = y_[::10]

for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(img[i].reshape(28,28),cmap='gray')
    plt.title('预测数据：%d'%(yimg[i])+'\n真实数据：%d'%(img1[i]))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.show()

'''import matplotlib.ticker as ticker
fig=plt.figure()
ax = fig.add_subplot(111)
ax.yaxis.set_major_locator(ticker.NullLocator())'''

