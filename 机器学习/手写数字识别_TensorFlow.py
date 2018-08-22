# coding:utf-8
import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt

# from sklearn.neighbors import KNeighborsClassifier

# 1、数据读取
data = plt.imread(r'./data/0/0_1.bmp')
# plt.imshow(data)
# plt.show()
# x_tain=[]
# for i in range(1,501):
#     x_tain.append(plt.imread('./data/0/0_%d.bmp'%(i)))

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
x_tain,y_tain,x_test= np.array(x_tain),np.array(y_tain),np.array(x_test)
y_tain = np.array(pd.get_dummies(y_tain))
# y_pred = pd.DataFrame(y_test)
# y_pred.to_csv('test/y_test.csv')
y_test = pd.get_dummies(y_test)
index,column = x_tain.shape
print(y_tain.shape)
print(index,column)


# 下面我们建立tensorflow模型
import tensorflow as tf
# 定义两个tensor张量
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

# 第一层输入层权值和偏置值
# w1 = tf.Variable(tf.zeros([784,100]))
# b1 = tf.Variable(tf.zeros([1,100]))

# # 第二层权值和偏置值
# w2 = tf.Variable(tf.zeros([500,200]))
# b2 = tf.Variable(tf.zeros([1,1]))

# 第三层
# w3 = tf.Variable(tf.zeros([200,100]))
# b3 = tf.Variable(tf.zeros([1,1]))
# 第四层
w4 = tf.Variable(tf.zeros([784,10]))
b4 = tf.Variable(tf.zeros([1,10]))

# 网络层一
# y1 = tf.nn.relu(tf.matmul(x,w1)+b1)

#网络层二

# y2 = tf.nn.tanh(tf.matmul(y1,w2)+b2)

# 网络层三

# y3 = tf.nn.relu(tf.matmul(y2,w3)+b3)

# 网络层四

y4 = tf.nn.softmax(tf.matmul(x,w4)+b4)

# 损失函数 loss
# loss = -tf.reduce_mean(y*tf.log(y4))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y4))

#优化器梯度下降法
train_model = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#求准确率
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y4,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#开始训练模型：
init = tf.global_variables_initializer()

n = b_size =100
with tf.Session() as sess:
    sess.run(init)
    for epcho in range(101):
        for i in range(4500//n):
            X_train = x_tain[(i * n):(n * (i + 1)),:]
            Y_train = y_tain[(i * n):(n * (i + 1)),:]
            sess.run(train_model,feed_dict={x:x_tain,y:y_tain})
        if epcho%20==0:
            auc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            print('epcho-%d'%epcho,'auc-',auc)
    y_pred = sess.run(y4,feed_dict={x:x_test})
    sess.close()

# y_pred = pd.DataFrame(y_pred)
# y_pred.to_csv('test/y_pred.csv')
