# conding:utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras import optimizers
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model

# 数据准备
x_train =np.zeros((4500,28,28,1))
x_test =np.zeros((500,28,28,1))
y_train=[]
y_test=[]

for i in range(0,10):
    for j in range(1,501):
        if j < 451: #将数据保存到训练数据中
            x_train[(j-1)+(i*450),:,:,0]=plt.imread('./data/%d/%d_%d.bmp'%(i,i,j)) #reshape 可以降维也就是矩阵变化
            y_train.append(i)  #append 是读进来的数据进行存储的意思
        else: #保存到预测数据中
            x_test[(i*50)+(j-452),:,:,0]=plt.imread('./data/%d/%d_%d.bmp'%(i,i,j))
            y_test.append(i)
y_t = np.array(y_test).reshape(-1,1)
print(x_train.shape)
# x_train = np.array(x_train).reshape(450,28,28,1)
y_train = np.array(pd.get_dummies(y_train))
print(y_train.shape)
# x_test = np.array(x_test).reshape(50,28,28,1)
y_test = np.array(pd.get_dummies(y_test))


# 模型建立

model = Sequential()
# 第一层：
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu',padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

#第二层：
# model.add(Conv2D(64,(5,5),activation='relu',padding='same',data_format='channels_first'))
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.25))
model.add(Conv2D(32,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
# 2、全连接层和输出层：
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',#,'binary_crossentropy'
              optimizer=optimizers.Adadelta(lr=0.2, rho=0.95, epsilon=1e-06),#,'Adadelta'
              metrics=['accuracy'])

# 模型训练
model.fit(x_train,y_train,batch_size=128,epochs=35)
y_y = model.predict(x_test)
score = model.evaluate(x_test, y_test, verbose=0)
# 保存模型
# model.save('test/my_model.h5')
print(score)
# 模型导入
# model = load_model('test/my_model.h5')
# y_y = model.predict(x_test)
# y_s = np.argmax(y_y,axis=1).reshape(-1,1)
# score_pred = len((y_t-y_s)[(y_t-y_s)==0])/len(y_t)
# print('准确率：',score_pred)
# plt.figure(figsize=(12,6))
# plt.scatter(list(range(len(y_s))),y_s,c=y_t)
# xlabel = ['数字0','数字1','数字2','数组3','数字4','数字5','数字6','数字7','数字8','数字9']
# plt.yticks(range(10),xlabel)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
# plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
# plt.show()

