# coding=utf-8 
# @Time :2018/11/21 12:51
"""
leastsq拟合y=kx+b可视化
"""
# 【最小二乘法试验】
import numpy as np
from scipy.optimize import leastsq

# 采样点(Xi,Yi)
Xi = np.array([8.19, 2.72, 6.39, 8.71, 4.7, 2.66, 3.78])
Yi = np.array([7.01, 2.78, 6.47, 6.71, 4.1, 4.23, 4.05])

"""part 1"""


# 需要拟合的函数func及误差error
def func(p, x):
    k, b = p
    return k * x + b


def error(p, x, y):
    return func(p, x) - y  # x、y都是列表，故返回值也是个列表


p0 = [1, 2]

# 最小二乘法求k0、b0
Para = leastsq(error, p0, args=(Xi, Yi))  # 把error函数中除了p以外的参数打包到args中
k0, b0 = Para[0]
print("k0=", k0, '\n', "b0=", b0)

"""part 2"""


# 定义一个函数，用于计算在k、b已知时，∑((yi-(k*xi+b))**2)
def S(k, b):
    ErrorArray = np.zeros(k.shape)
    for x, y in zip(Xi, Yi):
        ErrorArray += (y - (k * x + b)) ** 2
    return ErrorArray


# 绘制ErrorArray最低点
from mayavi import mlab

# 画整个Error曲面
k, b = np.mgrid[k0 - 1:k0 + 1:10j, b0 - 1:b0 + 1:10j]
Err = S(k, b)
face = mlab.surf(k, b, Err / 500.0, warp_scale=1)
mlab.axes(xlabel='k', ylabel='b', zlabel='Error')
mlab.outline(face)

# 画最低点
MinErr = S(k0, b0)
mlab.points3d(k0, b0, MinErr / 500.0, scale_factor=0.1, color=(0.5, 0.5, 0.5))
mlab.show()
