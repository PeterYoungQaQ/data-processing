# coding=utf-8
from numpy import *
import operator
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt


# 导入特征数据
def file2matrix(filename):
    fr = open(filename)
    contain = fr.readlines()  # 读取文件的所有内容
    count = len(contain)
    returnMat = zeros((count, 3))
    classLabeiVector = []
    index = 0
    for line in contain:
        line = line.strip()  # 截取所有的回车字符
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]  # 选组前三个元素，存储在特征矩阵中
        classLabeiVector.append(listFromLine[-1])  # 将列表的最后一列存储到向量classLabelVector中
        index += 1

    # 将列表的最后一列由字符串转化为数字，便于以后的计算
    # 列表的最后一列是约会信息，是字符串
    diceClassLabel = Counter(classLabeiVector)
    classLabel = []
    kind = list(diceClassLabel)
    for item in classLabeiVector:
        if item == kind[0]:
            item = 1
        elif item == kind[1]:
            item = 2
        else:
            item = 3
        classLabel.append(item)
    return returnMat, classLabel


# 绘图（可以直观的表示出各特征对分类结果的影响程度）
# datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1],
#            15.0 * array(datingLabels), 15.0 * array(datingLabels))
# plt.show()

# 归一化数据，保证特征等权重
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))  # 建立与dataSet结构一样的矩阵
    m = dataSet.shape[0]
    for i in range(1, m):
        normDataSet[i, :] = (dataSet[i, :] - minVals) / ranges
    return normDataSet, ranges, minVals


# KNN算法
def classify(input, dataSet, label, k):
    global classes
    dataSize = dataSet.shape[0]
    # 计算欧式距离
    diff = tile(input, (dataSize, 1)) - dataSet
    sqdiff = diff ** 2
    squareDist = sum(sqdiff, axis=1)  # 行向量分别相加，从而得到新的一个行向量
    dist = squareDist ** 0.5

    # 对距离进行排序
    sortedDistIndex = argsort(dist)  # argsort()根据元素的值从大到小对元素进行排序，返回下标
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


# 测试(选取10%测试)
def datingTest():
    rate = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    testNum = int(m * rate)
    errorCount = 0.0
    for i in range(1, testNum):
        classifyResult = classify(normMat[i, :], normMat[testNum:m, :],
                                  datingLabels[testNum:m], 3)

        print("分类后的结果为:,", classifyResult)
        print("原结果为：", datingLabels[i])
        if classifyResult != datingLabels[i]:
            errorCount += 1.0
    print("误分率为:", (errorCount / float(testNum)))


# datingTest()


# 预测函数
def classifyPerson():
    resultList = ['一点也不喜欢', '有一丢丢喜欢', '灰常喜欢']
    percentTats = float(input("玩视频所占的时间比?"))
    miles = float(input("每年获得的飞行常客里程数?"))
    iceCream = float(input("每周所消费的冰淇淋公升数?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([miles, percentTats, iceCream])
    classifierResult = classify((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你对这个人的喜欢程度:", resultList[classifierResult - 1])


classifyPerson()
