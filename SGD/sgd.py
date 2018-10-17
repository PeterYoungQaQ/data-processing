# coding=utf-8
"""
date : 2018-10-16
线性回归模型、GD算法、SGD算法的优化测试
具体公式参考: https://www.cnblogs.com/maybe2030/p/5089753.html
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 产生数据同时绘制图
np.random.seed(66)
# linspace(start, stop, num=50,
# endpoint=True, retstep=False, dtype=None)
# 可以认为是创建等差序列。这里刚好都是取整数
X = np.linspace(0, 50, 50)
# 在执行y = 2*x + 5的基础上加上噪声
y = 2 * X + 5 + np.random.rand(50) * 10
# 转换为二维数组
X.shape = (50, 1)
y.shape = (50, 1)


# plt.figure(figsize=(10, 10))
# plt.scatter(X, y,marker="p")
# plt.show()

# 训练模型
lr = LinearRegression()
lr.fit(X, y)
plt.figure(figsize=(10, 10))
plt.scatter(X, y, marker="o")
plt.plot(X, lr.predict(X), color='red', linewidth=3)
# plt.show()


# 梯度下降
def gradient_descent():
    X = np.linspace(0, 50, 50)
    X.shape = (50, 1)
    # 终止条件
    loop_max = 25000  # 最大循环次数
    epsilon = 21  # 误差阈值
    # 参数
    theta = np.random.rand(2, 1)  # 线性模型的系数，初始化为小随机数
    learning_rate = 0.001  # 学习率
    # 增加全1列
    X = np.hstack([np.ones((50, 1)), X])

    for i in range(loop_max):
        # dot()返回的是两个数组的点积
        grad = np.dot(X.T, (np.dot(X, theta) - y)) / X.shape[0]
        # 更新theta
        theta = theta - learning_rate * grad
        # 计算更新后的误差,(实际上求的是欧氏距离）
        error = np.linalg.norm(np.dot(X, theta) - y)
        # 输出当前的更新次数和误差
        print("The number of update is %d. The current error is %f" % (i, error))
        # 误差小于阈值时退出循环
        if error < epsilon:
            break
    # 绘制拟合的曲线
    plt.figure(figsize=(10, 10))
    plt.scatter(X[:, 1], y)
    plt.plot(X[:, 1], np.dot(X, theta), color='red', linewidth=3)
    plt.show()


# gradient_descent()

# 随机梯度下降
def stochastic_gradient_descent():
    # 还原参数theta，其他参数复用梯度下降
    theta = np.random.rand(1, 2)
    X = np.linspace(0, 50, 50)
    X.shape = (50, 1)

    # 终止条件
    loop_max = 35000  # 最大循环次数
    epsilon = 21  # 误差阈值
    learning_rate = 0.001  # 学习率
    # 指定每次更新使用的数据量
    batch_size = 10
    for i in range(loop_max):
        # 随机样本的列索引
        idxs = np.random.randint(0, X.shape[0], size=batch_size)
        # 随机样本
        tmp_X = X.take(idxs, axis=0)
        tmp_y = y.take(idxs, axis=0)
        # 计算梯度
        grad = np.dot(tmp_X.T, (np.dot(tmp_X, theta) - tmp_y)) / tmp_X.shape[0]
        # 更新theta
        theta = theta - learning_rate * grad
        # 计算更新后的误差
        error = np.linalg.norm(np.dot(X, theta) - y)
        # 输出当前的更新次数和误差
        print("The number of update is %d. The current error is %f"%(i,error))
        # 误差小于阈值时退出循环
        if error < epsilon:
            break
    # 绘制拟合的曲线
    plt.figure(figsize=(5, 5))
    plt.scatter(X[:, 0], y)
    plt.plot(X[:, 0], np.dot(X, theta), color='red', linewidth=3)
    plt.show()


# stochastic_gradient_descent()
