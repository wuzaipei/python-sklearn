# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# one_hot数据的读取
label = pd.read_csv(r'./test/y_pred.csv',header=0,index_col=0)
y_test = pd.read_csv(r'./test/y_test.csv',header=0,index_col=0)
label = label.values
print(label.shape)

x = np.zeros([len(label[:,0])]).reshape(-1,1)
print(x.shape)

n,m = label.shape

x = np.argmax(label,axis=1).reshape(-1,1)
print(np.max(x))
y = np.array(y_test.values).reshape(-1,1)
print(np.max(y))
source = len((x-y)[(x-y)==0])/len(x)

plt.figure('tensorflow-手写数字',figsize=(12,6))
plt.scatter(list(range(len(x))),x,c=y,label='source={0}'.format(source))
font_size = {'size':15}
plt.title('one_hot-label',font_size)
plt.xlabel('第i个数字',font_size)
plt.ylabel('数字类别',font_size)
plt.legend(loc='upper left')
plt.axis([0,530,0,10])
xlabel = ['数字0','数字1','数字2','数组3','数字4','数字5','数字6','数字7','数字8','数字9']
plt.xticks(range(0,501,100),['第0个','第100个','第200个','第300个','第400个','第500个'])
plt.yticks(range(10),xlabel)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.show()

