import numpy as np
import matplotlib.pyplot as plt
import time
start = time.clock()
x = np.linspace(-1,1,1000)

f = lambda x: (1-x**2)**0.5 #匿名函数

plt.figure(figsize=(6,6))
plt.plot(x,f(x),x,-f(x))
# plt.show()

# 积分计算圆周率
import scipy.integrate as inv

f_y = inv.quad(f,-1,1)  # 数值积分
pi_1 = f_y[0]*2 # 求pi
print(pi_1)

import scipy.io as spio

nd = np.random.randint(0,150,size=10)
# spio.savemat('nd',{'data':nd}) #scipy 的文件数据保存 二进制文件保存
# 读取：
# date = np.load('nd')['date']

# 读取图片

import scipy.misc as misc
plt.figure()
img = misc.imread('./data/0/0_1.bmp')
plt.imshow(img,cmap='gray')
plt.show()

# 高斯滤波

import scipy.ndimage as gs
# gs.gaussian_filter()

print('运行时间为%s秒'%(time.clock()-start))


