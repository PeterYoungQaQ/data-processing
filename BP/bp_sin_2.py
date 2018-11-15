# coding=utf-8 
# @Time :2018/11/10 15:01
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mal
from mpl_toolkits.mplot3d import Axes3D

x = np.linspace(0, 2 * math.pi, 100)
x_1 = np.linspace(0, 2 * math.pi, 100)
x_size = x.size
y = np.zeros((x_size, 1))
# print(y.size)
for i in range(x_size):
    y[i] = math.sin(x[i]) * math.sin(x_1[i])
x_n = x.tolist()
x_1_n = x_1.tolist()
x_r_m = x_n + x_1_n

x_r_n = np.array(x_r_m, dtype=float)
x_r = x_r_n.reshape(2, 100)




def sigmoid(x_):
    y_ = 1 / (1 + np.exp(-x_))
    return y_


def sinx_f():
    global e
    hidesize = 30
    W1 = np.random.random((hidesize, 2))  # 输入层与隐层之间的权重
    B1 = np.random.random((hidesize, 2))  # 隐含层神经元的阈值
    W2 = np.random.random((2, hidesize))  # 隐含层与输出层之间的权重
    B2 = np.random.random((1, 1))  # 输出层神经元的阈值
    threshold = 0.005
    max_steps = 1001
    E = np.zeros((max_steps, 1))  # 误差随迭代次数的变化
    Y = np.zeros((x_size, 1))  # 模型的输出结果
    for k in range(max_steps):
        temp = 0
        for i in range(x_size):
            hide_in_1 = (np.dot(x_r[0][i], W1[:, 0])) - B1[:, 0]  # 隐含层输入数据
            hide_in_2 = (np.dot(x_r[1][i], W1[:, 1])) - B1[:, 1]
            # print(x[i])
            hide_out = np.zeros((hidesize, 2))  # 隐含层的输出数据
            for j in range(hidesize):
                # print("第{}个的值是{}".format(j,hide_in[j]))
                # print(j,sigmoid(j))
                hide_out[j][0] = sigmoid(hide_in_1[j])
                hide_out[j][1] = sigmoid(hide_in_2[j])
                # print("第{}个的值是{}".format(j, hide_out[j]))


            y_out_1 = np.dot(W2[0], hide_out[:, 0])   # 模型输出
            y_out_2 = np.dot(W2[1], hide_out[:, 1])
            y_out = y_out_1 + y_out_2 - B2
            Y[i] = y_out

            e = y_out - y[i]  # 模型输出减去实际结果。得出误差
            # 反馈，修改参数
            dB2 = -1 * threshold * e
            dW2 = e * threshold * np.transpose(hide_out)
            dB1 = np.zeros((hidesize, 2))
            for j in range(hidesize):
                dB1[j][0] = np.dot(np.dot(W2[0][j], sigmoid(hide_in_1[j])),
                                (1 - sigmoid(hide_in_1[j])) * (-1) * e * threshold)
                dB1[j][1] = np.dot(np.dot(W2[1][j], sigmoid(hide_in_2[j])),
                                   (1 - sigmoid(hide_in_2[j])) * (-1) * e * threshold)
            dW1 = np.zeros((hidesize, 2))

            for j in range(hidesize):
                dW1[j][0] = np.dot(np.dot(W2[0][j], sigmoid(hide_in_1[j])),
                                (1 - sigmoid(hide_in_1[j])) * x_r[0][i] * e * threshold)
                dW1[j][1] = np.dot(np.dot(W2[1][j], sigmoid(hide_in_2[j])),
                                   (1 - sigmoid(hide_in_2[j])) * x_r[1][i] * e * threshold)
            W1 = W1 - dW1
            B1 = B1 - dB1
            W2 = W2 - dW2
            B2 = B2 - dB2
            temp = temp + abs(e)

        E[k] = temp

        if k % 100 == 0:
            print(k)
            print(e)
    return Y, E


Y_value_m, opp = sinx_f()
Y_value = Y_value_m.tolist()
t = len(Y_value)
Y_value_new = []
for i in range(t):
    m = (Y_value[i])[0]
    Y_value_new.append(m)
Y_value_new = list(map(float, Y_value_new))

rea_ans = []
for i in range(x.size):
    p = math.sin(x[i]) * math.sin(x_1[i])
    rea_ans.append(p)
rea_ans = list(map(float, rea_ans))

mal.rcParams['legend.fontsize'] = 10
fig = plt.figure()
asd = fig.gca(projection='3d')
tim = np.linspace(0, 100, 101)
# plt.plot(tim, opp, color='b')
asd.plot(x, x_1, Y_value_new, label="Predict", color='b')
asd.plot(x, x_1, rea_ans,label="Real", color='r')
plt.legend()
plt.show()