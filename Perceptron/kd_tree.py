# coding=utf-8 
# @Time :2018/11/27 18:40
"""
具体代码和详细说明参考：
https://www.cnblogs.com/bambipai/p/8436703.html
"""

import numpy as np
from numpy import array


class decisionnode:
    def __init__(self, value=None, col=None, rb=None, lb=None):
        self.value = value
        self.col = col
        self.rb = rb
        self.lb = lb


# 读取数据并将数据转换为矩阵形式
def readdata(filename):
    data = open(filename).readlines()
    x = []
    for line in data:
        line = line.strip().split('\t')
        x_i = []
        for num in line:
            num = float(num)
            x_i.append(num)
        x.append(x_i)
    x = array(x)
    return x


# 求序列的中值
def median(x):
    n = len(x)
    x = list(x)
    x_order = sorted(x)
    return x_order[n // 2], x.index(x_order[n // 2])


# 以j列的中值划分数据，左小右大，j=节点深度%列数
def buildtree(x, j=0):
    rb = []
    lb = []
    m, n = x.shape
    if m == 0:
        return None
    edge, row = median(x[:j].copy())
    for i in range(m):
        if x[i][j] > edge:
            rb.append(i)
        if x[i][j] < edge:
            lb.append(i)
    rb_x = x[rb, :]
    lb_x = x[lb, :]
    rightBranch = buildtree(rb_x, (j + 1) % n)
    leftBranch = buildtree(lb_x, (j + 1) % n)
    return decisionnode(x[row, :], j, rightBranch, leftBranch)


# 搜索树：nearestPoint,nearestValue均为全局变量
def traveltree(node, point):
    global nearestPoint, nearestValue
    if node is None:
        return
    print(node.value)
    print('---')
    col = node.col
    if point[col] > node.value[col]:
        traveltree(node.rb, point)
    if point[col] < node.value[col]:
        traveltree(node.lb, point)
    dis = dist(node.value, point)
    print(dis)
    if dis < nearestValue:
        nearestPoint = node
    nearestValue = dis
    # print('nearestPoint,nearestValue' % (nearestPoint,nearestValue))
    if node.rb is not None or node.lb is not None:
        if abs(point[node.col] - node.value[node.col]) < nearestValue:
            if point[node.col] < node.value[node.col]:
                traveltree(node.rb, point)
            if point[node.col] > node.value[node.col]:
                traveltree(node.lb, point)


def searchtree(tree, aim):
    global nearestPoint, nearestValue
    # nearestPoint=None
    nearestValue = float('inf')
    traveltree(tree, aim)
    return nearestPoint


def dist(x1, x2):  # 欧式距离的计算
    return ((np.array(x1) - np.array(x2)) ** 2).sum() ** 0.5
