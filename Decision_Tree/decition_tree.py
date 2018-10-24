# coding=utf-8
"""
    决策树和KNN一样，都是处理分类问题的算法
    在机器学习这个层面，将所要处理的数据看做是树的根，相应的选取数据的特征作为一个个节点（决策点），
每次选取一个节点将数据集分为不同的数据子集，可以看成对树进行分支，这里体现出了决策，
直到最后无法可分停止，也就是分支上的数据为同一类型，可以想象一次次划分之后由根延伸出了许多分支，形象的说就是一棵树。
　　在机器学习中，决策树是一个预测模型，它代表的是对象属性与对象值之间的一种映射关系，
我们可以利用决策树发现数据内部所蕴含的知识
"""

from math import log
import operator
import matplotlib.pyplot as plt

"""
    在划分数据集之前之后信息发生的变化称为信息增益，计算每个特征值划分数据集获得的信息增益，
获得信息增益最高的特征就是最好的选择。集合信息的度量方式称为香农熵或者简称为熵，熵定义为信息的期望值，
xi的信息可定义为：L(xi) = -log(p(xi)),其中p(xi)是选择该分类的概率。
    熵越高，表明混合的数据越多，则可以在数据集中添加更多的分类
"""


# 计算香农熵（为float类型）
def calShang(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}  # 创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 认为数据的最后一个部分是它的标签
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 输入数据集测试
def creatDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'yes'],
               [0, 1, 'no'],
               [0, 1, 'no'],
               [1, 1, 'yes'],
               [1, 0, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'yes'],
               [1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'yes'],
               [0, 0, 'no'],
               [0, 1, 'no'],
               [0, 0, 'no'],
               [0, 1, 'no'],
               [0, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['good', 'bad']
    return dataSet, labels


# 测试
# myData, labels = creatDataSet()
# print("原数据为:", myData)
# print("标签为:", labels)
# shang = calShang(myData)
# print("香农熵为:", shang)

# 划分数据集（以指定特征将数据进行划分）
# 传入待划分的数据集、划分数据集的特征以及需要返回的特征的值
def splitDataSet(dataSet, feature, value):
    newDataSet = []
    for featVec in dataSet:
        if featVec[feature] == value:
            reducedFeatVec = featVec[:feature]
            reducedFeatVec.extend(featVec[feature + 1:])
            newDataSet.append(reducedFeatVec)
    return newDataSet


# 测试
# myData, labels = creatDataSet()
# print("原数据为：", myData)
# print("标签为：", labels)
# split = splitDataSet(myData, 0, 1)
# print("划分后的结果为:", split)


# 选择最好的划分方式
# (选取每个特征划分数据集，从中选取信息增益最大的作为最优划分)
# 在这里体现了信息增益的概念
def chooseBest(dataSet):
    global bestFeature
    featNum = len(dataSet[0]) - 1
    baseEntropy = calShang(dataSet)
    bestInforGain = 0.0

    for i in range(featNum):
        featList = [example[i] for example in dataSet]  # 列表
        uniqueFeat = set(featList)  # 得到每个特征中所含的不同元素
        newEntropy = 0.0
        for value in uniqueFeat:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShang(subDataSet)
        inforGain = baseEntropy - newEntropy
        if inforGain > bestInforGain:
            bestInforGain = inforGain
            bestFeature = i  # 第i个特征是最有利于划分的特征
    return bestFeature


# myData, labels = creatDataSet()
# best = chooseBest(myData)
# print(best)

"""
    第一次划分之后，可以将划分的数据继续向下传递，
如果将每一个划分的数据看成是原数据集，那么之后的每
一次划分都可以看成是和第一次划分相同的过程，据此
我们可以采用递归的原则处理数据集。递归结束的条件
是：程序遍历完所有划分数据集的属性，或者每个分支
下的所有实例都有相同的分类。
"""


# 返回出现次数最多的分类名称
def majorClass(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 降序排序
    sortedClassCount = \
        sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 创建树
def creatTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # 如果只有一层。就是不需要区分的时候直接返回
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 如果最终只剩下一个属性，说明分类结束了，获得最好的分类结果
    if len(dataSet[0]) == 1:
        return majorClass(classList)
    bestFeat = chooseBest(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # 把这一次的数据清空
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = \
            creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


# 测试
# myData, labels = creatDataSet()
# mytree = creatTree(myData, labels)
# print(mytree)

# coding=utf-8


# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # fc 应该是颜色深浅
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # centerPt 箭头指向坐标， parentPt 箭头终点坐标
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotTree.totalW = float(getNumLeafs(inTree))  # 储存树的宽度
    plotTree.totalD = float(getTreeDepth(inTree))  # 储存树的深度
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


# createPlot()


# 定义两个新函数，来获取叶节点的数目和树的层数
def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys())  # dict.keys() 返回字典的 keys
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 利用 type() 函数测试节点的数据类型是否为字典
        if type(secondDict[key]).__name__ == 'dict':  # 如果模块是被导入，__name__的值为模块名字
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):  # 计算遍历过程总遇到判断节点的个数
    maxDepth = 0
    firstSides = list(myTree.keys())  # dict.keys() 返回字典的 keys
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):  # 在父子节点间填充文本信息
    # 计算父节点和子节点的中间位置
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):  # 计算树的宽与高
    numLeafs = getNumLeafs(myTree)
    firstSides = list(myTree.keys())  # dict.keys() 返回字典的 keys
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 /
              plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  # 标记子节点属性值
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD  # 减少 y 偏移
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


# 决策树的分类函数，返回当前节点的分类标签
def classify(inputTree, featLabels, testVec):  # 传入的数据为dict类型
    global classLabel
    firstSides = list(inputTree.keys())
    firstStr = firstSides[0]  # 找到输入的第一个元素
    ##这里表明了python3和python2版本的差别，上述两行代码在2.7中为：firstStr = inputTree.key()[0]
    secondDict = inputTree[firstStr]  # 建一个dict
    # print(secondDict)
    featIndex = featLabels.index(firstStr)  # 找到在label中firstStr的下标
    for i in secondDict.keys():
        print(i)

    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:  # 判断一个变量是否为dict，直接type就好
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel  # 比较测试数据中的值和树上的值，最后得到节点


# 测试
# myData, labels = creatDataSet()
# print(labels)
# mytree = retrieveTree(0)
# print(mytree)
# classify = classify(mytree, labels, [1, 0])
# print(classify)

# 使用决策树预测隐形眼镜类型

def glasses():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels = ['ages', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = creatTree(lenses, lensesLabels)
    print(lensesTree)
    createPlot(lensesTree)


# glasses()
