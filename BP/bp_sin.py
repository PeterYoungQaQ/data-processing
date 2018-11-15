import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib as mal

x = np.linspace(-2 * math.pi, 2 * math.pi, 100)
x_1 = np.linspace(-2 * math.pi, 2 * math.pi, 100)
x_size = x.size
y = np.zeros((x_size, 1))
# print(y.size)
for i in range(x_size):
    y[i] = math.sin(x[i])
x_1_size = x_1.size
y_1 = np.zeros((x_size, 1))
# print(y.size)
for i in range(x_1_size):
    y_1[i] = math.sin(x_1[i])


def sigmoid(x_):
    y_ = 1 / (1 + np.exp(-x_))
    return y_


def sinx_f():
    global e
    hidesize = 10
    W1 = np.random.random((hidesize, 1))  # 输入层与隐层之间的权重
    B1 = np.random.random((hidesize, 1))  # 隐含层神经元的阈值
    W2 = np.random.random((1, hidesize))  # 隐含层与输出层之间的权重
    B2 = np.random.random((1, 1))  # 输出层神经元的阈值
    threshold = 0.005
    max_steps = 101
    E = np.zeros((max_steps, 1))  # 误差随迭代次数的变化
    Y = np.zeros((x_size, 1))  # 模型的输出结果
    for k in range(max_steps):
        temp = 0
        for i in range(x_size):
            hide_in = np.dot(x[i], W1) - B1  # 隐含层输入数据
            # print(x[i])
            hide_out = np.zeros((hidesize, 1))  # 隐含层的输出数据
            for j in range(hidesize):
                # print("第{}个的值是{}".format(j,hide_in[j]))
                # print(j,sigmoid(j))
                hide_out[j] = sigmoid(hide_in[j])
                # print("第{}个的值是{}".format(j, hide_out[j]))

            # print(hide_out[3])
            y_out = np.dot(W2, hide_out) - B2  # 模型输出

            Y[i] = y_out

            e = y_out - y[i]  # 模型输出减去实际结果。得出误差
            ##反馈，修改参数
            dB2 = -1 * threshold * e
            dW2 = e * threshold * np.transpose(hide_out)
            dB1 = np.zeros((hidesize, 1))
            for j in range(hidesize):
                dB1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])),
                                (1 - sigmoid(hide_in[j])) * (-1) * e * threshold)

            dW1 = np.zeros((hidesize, 1))

            for j in range(hidesize):
                dW1[j] = np.dot(np.dot(W2[0][j], sigmoid(hide_in[j])),
                                (1 - sigmoid(hide_in[j])) * x[i] * e * threshold)

            W1 = W1 - dW1
            B1 = B1 - dB1
            W2 = W2 - dW2
            B2 = B2 - dB2
            temp = temp + abs(e)

        E[k] = temp

        if k % 100 == 0:
            print(k)
            print(e)
    return Y


def sin_1_x():
    global e_1
    hidesize_1 = 10
    W1_1 = np.random.random((hidesize_1, 1))  # 输入层与隐层之间的权重
    B1_1 = np.random.random((hidesize_1, 1))  # 隐含层神经元的阈值
    W2_1 = np.random.random((1, hidesize_1))  # 隐含层与输出层之间的权重
    B2_1 = np.random.random((1, 1))  # 输出层神经元的阈值
    threshold_1 = 0.005
    max_steps_1 = 101
    E_1 = np.zeros((max_steps_1, 1))  # 误差随迭代次数的变化
    Y_1 = np.zeros((x_1_size, 1))
    for k in range(max_steps_1):
        temp_1 = 0
        for i in range(x_1_size):
            hide_in_1 = np.dot(x_1[i], W1_1) - B1_1  # 隐含层输入数据
            # print(x[i])
            hide_out_1 = np.zeros((hidesize_1, 1))  # 隐含层的输出数据
            for j in range(hidesize_1):
                # print("第{}个的值是{}".format(j,hide_in[j]))
                # print(j,sigmoid(j))
                hide_out_1[j] = sigmoid(hide_in_1[j])
                # print("第{}个的值是{}".format(j, hide_out[j]))

            # print(hide_out[3])
            y_1_out = np.dot(W2_1, hide_out_1) - B2_1  # 模型输出

            Y_1[i] = y_1_out

            e_1 = y_1_out - y_1[i]  # 模型输出减去实际结果。得出误差
            ##反馈，修改参数
            dB2_1 = -1 * threshold_1 * e_1
            dW2_1 = e_1 * threshold_1 * np.transpose(hide_out_1)
            dB1_1 = np.zeros((hidesize_1, 1))
            for j in range(hidesize_1):
                dB1_1[j] = np.dot(np.dot(W2_1[0][j], sigmoid(hide_in_1[j])),
                                (1 - sigmoid(hide_in_1[j])) * (-1) * e-1 * threshold_1)

            dW1_1 = np.zeros((hidesize_1, 1))

            for j in range(hidesize_1):
                dW1_1[j] = np.dot(np.dot(W2_1[0][j], sigmoid(hide_in_1[j])),
                                (1 - sigmoid(hide_in_1[j])) * x_1[i] * e_1 * threshold_1)

            W1_1 = W1_1 - dW1_1
            B1_1 = B1_1 - dB1_1
            W2_1 = W2_1 - dW2_1
            B2_1 = B2_1 - dB2_1
            temp_1 = temp_1 + abs(e_1)

        E_1[k] = temp_1

        if k % 100 == 0:
            print(k)
            print(e_1)
    return Y_1


Y_value = sinx_f().tolist()
Y_1_value = sin_1_x().tolist()
t = len(Y_value)
Y_value_new = []
Y_value_1_new = []
for i in range(t):
    m = (Y_value[i])[0]
    n = (Y_1_value[i])[0]
    Y_value_new.append(m)
    Y_value_1_new.append(n)
Y_value_new = list(map(float, Y_value_new))
Y_value_1_new = list(map(float, Y_value_1_new))

Y_ans = []
for i in range(t):
    ans = Y_value_new[i] * Y_value_1_new[i]
    Y_ans.append(ans)
Y_ans = list(map(float, Y_ans))


mal.rcParams['legend.fontsize'] = 10
fig = plt.figure()
asd = fig.gca(projection='3d')
asd.plot(x, x_1, Y_ans, color='b')
plt.legend()
plt.show()
