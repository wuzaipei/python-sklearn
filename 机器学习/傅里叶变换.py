import numpy as np
#fft 傅里叶转换 ifft 傅里叶反转
from numpy.fft import fft,ifft
from PIL import Image
cat = Image.open('C:\\Users\\wuzaipei\\Pictures\\Camera Roll\\地球.jpg')
print(type(cat))
cat.show()

def rgb2gray(rgb):
 return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# 2、数据类型的转换
# 之所以有负数，因为我们给的是int8，而int8的最大值为128，而我们的图片数据有0到255
# 但是我们为了把图片数据变归一，所以选int8
cat_data = np.fromstring(cat.tobytes(),dtype=np.int8)


# 傅里叶变换
cat_data1 = fft(cat_data) #这里面有虚数可以用傅里叶部分来把它转换成虚数表达
# print(cat_data1)

# 将傅里叶的数据去除低频的波，设置为0，比如提莫的耳朵
# 等于增加滤波的意思
# 进行滤除
#np.where(np.abs(cat_data<1e5,0,cat_data1))  #这行代码和下面这行代码是一样的操作
cat_data1[np.where(np.abs(cat_data1)<1e5)]=0

# 使用ifft进行反转傅里叶变换
cat_data_ifft = ifft(cat_data1)

cat_real = np.real(cat_data_ifft)  # 获取实部部分数据

# 去除小数部分
cat_data_result = np.int8(cat_real)
print(cat_data_result)

cat_img = Image.frombytes(size=cat.size,mode=cat.mode,data=cat_data_result)
cat_img.show()

