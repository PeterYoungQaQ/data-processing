# coding=utf-8

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")  # fc 应该是颜色深浅
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # centerPt 箭头指向坐标， parentPt 箭头终点坐标
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('decisionNode', (0.5, 0.1), (0.1, 0.5), decisionNode)  # U 这里指的是 utf 编码
    plotNode('leafNode', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


createPlot()
