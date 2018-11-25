# coding=utf-8 
# @Time :2018/11/25 11:02

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

training_set = np.array([[[3, 3], 1], [[4, 3], 1],
                         [[1, 1], -1], [[5, 2], -1]])  # 训练样本

# 矩阵a的长度为训练集样本数，类型为float
a = np.zeros(len(training_set), np.float)
b = 0.0  # 参数初始值为0
Gram = None
y = np.array(training_set[:, 1])  # y=[1 1 -1 -1]
x = np.empty((len(training_set), 2), np.float)  # x为4*2的矩阵
for i in range(len(training_set)):  # x=[[3., 3.], [4., 3.], [1., 1.], [5., 2.]]
    x[i] = training_set[i][0]
history = []  # history记录每次迭代结果


def cal_gram():
    """
    计算Gram矩阵
    :return:
    """
    g = np.empty((len(training_set), len(training_set)), np.int)
    for m in range(len(training_set)):
        for j in range(len(training_set)):
            g[m][j] = np.dot(training_set[m][0], training_set[j][0])
    return g


def update(new):
    """
    随机梯度下降更新参数,假设学习效率η为1
    :param new:
    :return:
    """
    global a, b
    a[new] += 1  # 根据误分类点更新参数
    b = b + 1 * y[new]
    history.append([np.dot(a * y, x), b])
    print(a, b)


# 计算yi(Gram*xi+b),用来判断是否是误分类点
def cal(i):
    global a, b, x, y
    res = np.dot(a * y, Gram[i])
    res = (res + b) * y[i]  # 返回
    return res


# 检查是否已经正确分类
def check():
    global a, b, x, y
    flag = False
    for i in range(len(training_set)):  # 遍历每一个点
        if cal(i) < 0:
            flag = True
            update(i)
    if not flag:  # 如果已经都正确分类
        w = np.dot(a * y, x)
        print("Result: w:" + str(w) + "b: " + str(b))
        return False
    return True


if __name__ == "__main__":
    Gram = cal_gram()
    for i in range(1000):
        if not check():
            break  # 如果已经正确分类则结束循环
    # 以下代码是将迭代过程可视化,数据来源于history
    # fig = plt.figure()
    # ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
    # line, = ax.plot([], [], 'g', lw=2)
    # label = ax.text([], [], '')
    #
    #
    # def init():
    #     line.set_data([], [])
    #     x, y, x_, y_ = [], [], [], []
    #     for p in training_set:
    #         if p[1] > 0:
    #             x.append(p[0][0])
    #             y.append(p[0][1])
    #         else:
    #             x_.append(p[0][0])
    #             y_.append(p[0][1])
    #     plt.plot(x, y, 'bo', x_, y_, 'rx')
    #     plt.axis([-6, 6, -6, 6])
    #     plt.grid(True)
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.title('PerceptronAlgorithm')
    #     return line, label
    #
    #
    # #  animation function. this is called sequentially
    #
    # def animate(i):
    #     global history, ax, line, label
    #     w = history[i][0]
    #     b = history[i][1]
    #     if w[1] == 0:
    #         return line, label
    #     x1 = -7.0
    #     y1 = -(b + w[0] * x1) / w[1]
    #     x2 = 7.0
    #     y2 = -(b + w[0] * x2) / w[1]
    #     line.set_data([x1, x2], [y1, y2])
    #     x1 = 0.0
    #     y1 = -(b + w[0] * x1) / w[1]
    #     label.set_text(str(history[i][0]) + ' ' + str(b))
    #     label.set_position([x1, y1])
    #     return line, label
    #
    #
    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                                frames=len(history), interval=1000, repeat=True,
    #                                blit=True)
    # plt.show()