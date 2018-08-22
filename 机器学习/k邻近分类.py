# coding:utf-8
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# 1、创造数据
data=pd.read_excel(r'data1.xlsx',header=None)
biaoq=data.columns
x_data = data[biaoq[:3]]
y_data = data[biaoq[3]]
x_tain =x_data[:900]
x_test = x_data[900:]
y_tain = y_data[:900]
y_test = y_data[900:]

# 2、构造分类器
knn = KNeighborsClassifier(n_neighbors=7)
# 3、训练模型
knn.fit(x_tain,y_tain)
# 4、模型预测
y_ = knn.predict(x_test)
# 5、准确率
gl= knn.score(x_test,y_test)
print(gl)
