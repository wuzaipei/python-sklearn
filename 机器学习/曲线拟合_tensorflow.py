# coding:utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Sigmoid import sigmoid

x_data = np.arange(-2*np.pi,2*np.pi,0.1).reshape(-1,1)
y_data = np.sin(x_data).reshape(-1,1)
# x_data = sigmoid(x_data)
# y_data = sigmoid(y_data)
print(x_data.shape,y_data.shape)

# 建立tensorflow模型
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])
# 首层
w = tf.Variable(tf.random_normal([1,10]))
b = tf.Variable(tf.zeros([1,10]))
# 中间层
w1 = tf.Variable(tf.random_normal([10,20]))
b1 = tf.Variable(tf.zeros([1,1]))
# 输出层
w2 = tf.Variable(tf.random_normal([20,1]))
b2 = tf.Variable(tf.zeros([1,1]))

y_pred = tf.matmul(x,w)+b
# 激活函数
y_pred_1 = tf.nn.tanh(y_pred)
yy  = tf.matmul(y_pred_1,w1)+b1
y_pred_ = tf.nn.tanh(yy)
y1 = tf.matmul(y_pred_,w2)+b2
y2 = tf.nn.tanh(y1)
#二次代价函数
loss = tf.reduce_mean(tf.square(y-y2))
# 训练方法：梯度下降法
train_model = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y2,1))#argmax返回一维张量中最大的值所在的位置
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
# 初始化变量
inint = tf.global_variables_initializer()
# 开始训练
with tf.Session() as sess:
    sess.run(inint)
    for i in range(10000):
        sess.run(train_model,feed_dict={x:x_data,y:y_data})
        if i%1000==0:
            auc = sess.run(accuracy,feed_dict={x:x_data,y:y_data})
            print('迭代次数：%d'%i,'auc:%d'%auc,' 损失函数(loss)：',sess.run(loss,feed_dict={x:x_data,y:y_data}))
    y_ = sess.run(y2,feed_dict={x:x_data})
    sess.close()

    plt.figure('tensorflow',figsize=(12,6))
    plt.scatter(x_data, y_data,label='sin(x)的值')
    plt.plot(x_data,y_,'r',linewidth=1,label='tensorflow拟合值')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
    plt.title('tensorflow实现y=sin(x)拟合')
    plt.xlabel('x-values',{'size':15})
    plt.ylabel('y-values-sin(x)',{'size':15})
    plt.legend(loc='upper right')
    plt.show()


