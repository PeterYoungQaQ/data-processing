# coding=utf-8 
# @Time :2018/11/20 22:35
"""
函数形为y=ax^2+bx+c 的最小二乘法的测试
"""

import numpy as np
from scipy.optimize import leastsq

Xi = np.array([0, 1, 2, 3, -1, -2, -3])
Yi = np.array([-1.21, 1.9, 3.2, 10.3, 2.2, 3.71, 8.7])


# 需要拟合的函数func及误差error
def func(p, x):
    a, b, c = p
    return a * x ** 2 + b * x + c


def error(p, x, y, s):
    print(s)
    return func(p, x) - y


# Test
p0 = [5, 2, 10]
# print(error(p0, Xi, Yi, 0))
s = "Test the number of iteration"
# 试验最小二乘法函数leastsq得调用几次error函数才能找到使得
# 均方误差之和最小的a~c

Para = leastsq(error, p0, args=(Xi, Yi, s))  # 把error函数中除了p以外的参数打包到args中
a, b, c = Para[0]
print("a=", a, '\n', "b=", b, "c=", c)

# 绘图
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(Xi, Yi, color="red", label="Sample Point", linewidths=3)
x = np.linspace(-5, 5, 1000)
y = a * x ** 2 + b * x + c

plt.plot(x, y, color="orange", label="Fitting Curve", linewidth=2)
plt.legend()
plt.show()
