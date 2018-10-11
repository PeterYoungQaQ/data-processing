# coding=utf-8
"""
    KNN(k-NearestNeighbor)是通过测量不同特征值之间的距离进行分类。
它的思路是：如果一个样本在特征空间中的k个最相似(即特征空间中最邻近)的样本中的大多数属于某一个类别，
则该样本也属于这个类别，其中K通常是不大于20的整数。
    KNN算法中，所选择的邻居都是已经正确分类的对象。
该方法在定类决策上只依据最邻近的一个或者几个样本的类别来决定待分样本所属的类别。

其算法的描述为：
1）计算测试数据与各个训练数据之间的距离；
2）按照距离的递增关系进行排序；
3）选取距离最小的K个点；
4）确定前K个点所在类别的出现频率；
5）返回前K个点中出现频率最高的类别作为测试数据的预测分类
"""

from numpy import *
import numpy as np
import operator


# 给出训练数据以及对应的类别
def createDataSet():
    group = np.array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 通过KNN进行分类
def classify(input, dataSet, label, k):
    # shape[0]就是数据排成一行表示
    global classes
    dataSize = dataSet.shape[0]

    # 计算欧式距离
    # numpy.tile(A,reps)
    # tile共有2个参数，A指待输入数组，reps则决定A重复的次数。
    # 整个函数用于重复数组A来构建新的数组。
    diff = np.tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    # 对距离进行排序,根据元素从大到小排序，返回下标
    sortedDistIndex = np.argsort(dist)

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        # 对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    # 选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes


def test():
    dataSet, labels = createDataSet()
    input = np.array([1.5, 0.1])
    K = 3
    output = classify(input, dataSet, labels, K)
    print("测试数据为:", input, "分类结果为：", output)


# test()
