import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import xlrd


#import xlrd 导入读取Excel的库

# 1、下面我们来读取原始数据使用pandas库中的read进行读取

data=pd.read_table('datingTestSet2.txt',header=None)  # header 设置是给她加上一个列标签好进行索引。

print(data.head())
