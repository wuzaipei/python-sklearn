# coding:utf-8
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
#生成数据
dot1= np.random.randn(20,2)+[-3,-3] #生成正态分布数据
dot2 = np.random.randn(20,2)+[2,3]
x_tain = np.r_[dot1,dot2]
y_tain = [0]*20+[1]*20  #这样y_tain 为40行2列数据

# 建立SVC模型
svc =SVC(kernel='linear')
svc.fit(x_tain,y_tain)
y_test = svc.predict(x_tain)

y_ = svc.coef_  #斜率
w = -y_[0,0]/y_[0,1]
j_ = svc.intercept_ #截距
ju_ = -j_[0]/y_[0,0]

plt.scatter(x_tain[:,0],x_tain[:,1],c=y_test)
x = np.arange(x_tain[:,0].min()-1,x_tain[:,0].max()+1,0.1)
plt.plot(x,w*x+ju_)
#获取支持向量
sv_=svc.support_vectors_
plt.scatter(sv_[:,0],sv_[:,1],s=100,c='red',alpha=0.3)
# 求解截距
sub = sv_[0]
uper_b = sub[1]- w* sub[0]
sub1=sv_[1]
uper_z = sub1[1]- w* sub1[0]
plt.plot(x,w*x+uper_b,c='r',ls='--')
plt.plot(x,w*x+uper_z,c='g',ls='--')
plt.show()
