import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
start = time.clock()
# 随机数
rng = np.random

# 参数
learning_rate = 0.01
#学习1000次
training_epochs = 2500
display_step = 20

train_X = np.linspace(0,10,num= 20)+np.random.randn(20)
train_Y = np.linspace(1,4,num = 20)
# y_2=np.linspace(1,4,num = 20)+np.random.randn(20)
# train_Y =np.array([y_1])
n_samples = 30


plt.subplot(1,2,1)
plt.scatter(train_X,train_Y)


print(train_Y.shape)
#使用之前的线性回归，将数据进行训练和学习

from sklearn.linear_model import LinearRegression

lrg = LinearRegression()

#训练数据
lrg.fit(train_X.reshape((20,1)),train_Y)

#声明预测数据
x_test = np.linspace(-2,12,num = 100).reshape((100,1))
#预测
y_ = lrg.predict(x_test)
plt.scatter(train_X,train_Y)

#输出预测结果
plt.plot(x_test,y_,c = 'green')

print(lrg.coef_,lrg.intercept_)


# tf Graph Input
#占位符，此时X，Y并没有赋值，训练的时候进行赋值
#通过feed这种方式进行赋值
#未知数据
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
#Variable变量，定义了，斜率和截距，成功求解
#weight，权重，x*weight = y
#斜率就是 weight
#Variable变量
# W ，b一开始蒙了一个数据
W= tf.Variable(rng.randn())
#b截距，bias 有偏差，就相当于有截距
b = tf.Variable(rng.randn())

# Construct a linear model
#f(x) = w*x + b
#预测值，train_Y原本的值
#并不是真实的，需要使用TensorFlow 进行学习，尽心处理
#函数f(x)
# 预测函数
y_pred = tf.add(tf.multiply(X, W), b)

# 均方误差，平均误差
#cost越小，数据越精确
# 损失函数，y_pred 预测函数（f(x) = w*x + b）
# Y 真实的数据，y值，创造20个点，
# (Y - y_pred)^2/样本量   平均最小二乘法
# cost越小，说明预测函数越精确---->w,b---->答案
cost = tf.reduce_mean(tf.pow(y_pred-Y, 2))

#算法
# 实现梯度下降算法的优化器
#learning_rate = 0.01
#minimize:最小化 进行梯度下降的条件
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
# 训练开始
plt.subplot(1,2,2)
with tf.Session() as sess:
    # 初始化
    sess.run(init)

    # 学习了20000 次
    # 训练所有数据 1000次for循环
    for epoch in range(training_epochs):
        # 执行20次
        # TensorFlow可以连续学习
        for (x, y) in zip(train_X, train_Y):
            # 每次for循环执行了梯度下降的算法
            #             条件cost值最小
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # 每执行50次显示运算结果
        if (epoch + 1) % 500 == 0:
            # cost 最小二乘法平方差之和的平局值
            #             cost = tf.reduce_sum(tf.pow(y_pred-Y, 2))/n_samples
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            #           f(x) = w*x + b---->w:weight;  b: bias(偏差)
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    # 算法优化结束
    print("Optimization Finished!")

    # 平均偏差，最终结果
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # 数据可视化
    plt.plot(train_X, train_Y, 'ro', label='原始数据')
    # f(x) = w*x + b
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='拟合线')
    plt.legend()

# 计算时间
end = time.clock()
print('一共用时：%s秒'%(end-start))

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 设置正常显示符号
plt.show()
print(lrg.coef_,lrg.intercept_)

sess.close()




