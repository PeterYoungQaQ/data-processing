# coding=utf-8 
# @Time :2018/11/14 19:27
"""
    二分类进行分析，在回归分析中需要一个函数可以接受所有的输入然后预测
出类别，假定用0和1分别表示两个类别，logistic函数曲线很像S型，故此我们
可以联系sigmoid函数：σ = 1/(1/(1+e-z))。为了实现logistic回归分类
器，我们可以在每个特征上乘以一个回归系数，将所有的乘积相加，将和值代入
sigmoid函数中，得到一个范围为0-1之间的数，如果该数值大于0.5则被归入
1类，否则被归为0类。
    基于之前的分析，需要找到回归系数，首先我们可以将sigmoid函数的输入
形式记为：z = w0x0 + w1x1 +...+wnxn,其中x为输入数据，相应的w就是我
们要求的系数，为了求得最佳系数，结合最优化理论，我们可以选取梯度上升法
优化算法。梯度上升法的基本思想是:要找到函数的最大值，最好的方法是沿着
该函数的梯度方向寻找。
"""

from numpy import *
import math
import matplotlib.pyplot as plt


# 导入数据
def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()  # 将文本中的每行中的字符一个个分开，编程list
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


# 定义sigmoid函数
def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 梯度上升方法求出回归系数
def gradAscent(datas, labels):
    dataMat = mat(datas)
    labelMat = mat(labels).transpose()
    m, n = shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weight = ones((n, 1))
    for item in range(maxCycles):
        h = sigmoid(dataMat * weight)
        error = (labelMat - h)  # 注意labelMat中的元素的数据类型为int
        weight = weight + alpha * dataMat.transpose() * error
    return weight


# data, label = loadDataSet()
# print(gradAscent(data, label))

# 求出回归系数之后，就确定了不同数据类别之间的分隔线，
# 为了便于理解，可以画出那条线
def plotBestFit(weight):
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcode1 = []
    ycode1 = []
    xcode2 = []
    ycode2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcode1.append(dataArr[i, 1])
            ycode1.append(dataArr[i, 2])
        else:
            xcode2.append(dataArr[i, 1])
            ycode2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcode1, ycode1, s=30, c='red', marker='s')
    ax.scatter(xcode2, ycode2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weight[0] - weight[1] * x) / weight[2]
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()


# data, label = loadDataSet()
# weights = gradAscent(data, label)
# plotBestFit(weights.getA())

# 改进的梯度上升法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = shape(dataMatrix)
    weights = ones(n)  # initialize to all ones
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights


#测试
data,label = loadDataSet()
weights = stocGradAscent1(array(data),label)
plotBestFit(weights)