# coding=utf-8 
# @Time :2018/11/25 11:54

from __future__ import division
import random
import numpy as np
import matplotlib.pyplot as plt


def sign(v):
    if v >= 0:
        return 1
    else:
        return -1


def train(train_num, train_datas, lr):
    w = 0.0
    b = 0
    datas_len = len(train_datas)
    alpha = [0 for i in range(datas_len)]
    train_array = np.array(train_datas)
    gram = np.matmul(train_array[:, 0:-1], train_array[:, 0:-1].T)
    for idx in range(train_num):
        tmp = 0
        i = random.randint(0, datas_len - 1)
        yi = train_array[i, -1]
        for j in range(datas_len):
            tmp += alpha[j] * train_array[j, -1] * gram[i, j]
        tmp += b
        if yi * tmp <= 0:
            alpha[i] = alpha[i] + lr
            b = b + lr * yi
    for i in range(datas_len):
        w += alpha[i] * train_array[i, 0:-1] * train_array[i, -1]
    return w, b, alpha, gram


def plot_points(train_datas, w, b):
    plt.figure()
    x1 = np.linspace(0, 8, 100)
    x2 = (-b - w[0] * x1) / (w[1] + 1e-10)
    plt.plot(x1, x2, color='r', label='y1 data')
    datas_len = len(train_datas)
    for i in range(datas_len):
        if train_datas[i][-1] == 1:
            plt.scatter(train_datas[i][0], train_datas[i][1], s=50)
        else:
            plt.scatter(train_datas[i][0], train_datas[i][1], marker='x', s=50)
    plt.show()


if __name__ == '__main__':
    train_data1 = [[1, 3, 1], [2, 2, 1], [3, 8, 1], [2, 6, 1]]  # 正样本
    train_data2 = [[2, 1, -1], [4, 1, -1], [6, 2, -1], [7, 3, -1]]  # 负样本
    train_datas = train_data1 + train_data2  # 样本集
    w, b, alpha, gram = train(train_num=500, train_datas=train_datas, lr=0.01)
    print(w, b)
    print(alpha)
    plot_points(train_datas, w, b)
