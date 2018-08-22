# coding:utf-8

def sigmoid(x):
    import numpy as np
    x = np.array(x)
    n,m = x.shape
    data = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            data[i,j] = 1.0 / (1 + np.exp(-float(x[i,j])))
    return data
