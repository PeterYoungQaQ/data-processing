# coding=utf-8
from numpy import *
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt


def loadData(fileName):
    data = []
    fr = open(fileName)
    for line in fr.readlines():
        curline = line.strip().split('\t')
        frline = list(map(float, curline))
        data.append(frline)
    return data


# test
# a = mat(loadData("testSet.txt"))
# print(a)

# 计算欧式距离
def distElud(vecA, vecB):
    return sqrt(sum(power((vecA - vecB), 2)))


# 初始化聚类中心
def randCent(dataSet, k):
    n = shape(dataSet)[1]
    center = mat(zeros((k, n)))
    for j in range(n):
        rangeJ = float(max(dataSet[:, j]) - min(dataSet[:, j]))
        center[:, j] = min(dataSet[:, j]) + rangeJ * random.rand(k, 1)
    return center


# test
# a = mat(loadData("testSet.txt"))
# n = 3
# b = randCent(a, n)
# print(b)


def kMeans(dataSet, k, dist=distElud, createCent=randCent):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 2)))
    center = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf
            minIndex = -1
            for j in range(k):
                distJI = dist(dataSet[i, :], center[j, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:  # 判断是否收敛
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        print(center)
        for cent in range(k):  # 更新聚类中心
            dataCent = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            center[cent, :] = mean(dataCent, axis=0)  # axis是普通的将每一列相加，而axis=1表示的是将向量的每一行进行相加
    return center, clusterAssment


# # test
# dataSet = mat(loadData("testSet.txt"))
# k = 4
# a = kMeans(dataSet, k)
# print(a)


def showCluster(dataSet, k, centroids, clusterAssment):
    numSamples, dim = dataSet.shape  # numSample - 样例数量  dim - 数据的维度
    if dim != 2:
        print("Sorry! I can not draw because the dimension os your data is not 2!")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("Sorry! Your k is too large! Please contact Zouxy")
        return 1

    # draw all samples
    for i in range(numSamples):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']

    # draw the centroids
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i], ms=12.0)
    plt.show()


# 绘图测试
# dataSet = mat(loadData("testSet.txt"))
# k = 4
# centroids, clusterAssment = kMeans(dataSet, k)
# showCluster(dataSet, k, centroids, clusterAssment)
