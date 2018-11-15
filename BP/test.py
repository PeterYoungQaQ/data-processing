# coding=utf-8 
# @Time :2018/11/10 15:32
import numpy as np
import math

x = np.linspace(-2 * math.pi, 2 * math.pi, 100)
x_1 = np.linspace(-2 * math.pi, 2 * math.pi, 100)
x_size = x.size
y = np.zeros((x_size, 1))
# print(y.size)
for i in range(x_size):
    y[i] = math.sin(x[i]) * math.sin(x_1[i])

x_n = x.tolist()
x_1_n = x_1.tolist()
x_r = x_n + x_1_n

x_r_n = np.array(x_r, dtype=float)
x_r_m = x_r_n.reshape(2, 100)
x_r = x_r_m.tolist()
print(x_r_m.shape)
