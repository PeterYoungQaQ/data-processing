# coding=utf-8 
# @Time :2018/11/24 17:28
"""
    之前的模型都是通过训练多个学习器后分别得到结果后整合为最终结果，
整合的过程为投票、求平均、求加权平均等统计方法。那为什么不把每个学习
器得到的结果作为特征进行训练(Blend)，再预测出最后的结果。
    需要将训练数据分为两部分。第1部分用于训练多个基学习器。
    第2部分（hold-out set）用于训练blender。blender
的输入为第2部分数据在第一部分数据训练好的多个模型的预测结果。
    实际上可以训练多个blender（如一个Logistic回归，另一个
Ramdomforest）实现这个思想的诀窍是将训练集分成三份，第一份
用于训练多个基学习器，第二份用于训练第二个层（使用第一个层的
预测器进行的预测作为输入），第三份用于训练第三层（使用第二层的
预测器进行的预测作为输入）。
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from brew.base import Ensemble, EnsembleClassifier
from brew.stacking.stacker import EnsembleStack, EnsembleStackClassifier
from brew.combination.combiner import Combiner

from mlxtend.data import iris_data
from mlxtend.plotting import plot_decision_regions

# 初始化分类器
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)

# 创建不同分类器之间的合作共享机制
ensemble = Ensemble([clf1, clf2, clf3])
eclf = EnsembleClassifier(ensemble=ensemble, combiner=Combiner('mean'))

# 创建不同层之间的合作共享机制
layer_1 = Ensemble([clf1, clf2, clf3])
layer_2 = Ensemble([sklearn.clone(clf1)])

stack = EnsembleStack(cv=3)

stack.add_layer(layer_1)
stack.add_layer(layer_2)

sclf = EnsembleStackClassifier(stack)

clf_list = [clf1, clf2, clf3, eclf, sclf]
lbl_list = ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble', 'Stacking']

# 初始化和载入数据

X, y = iris_data()
X = X[:, [0, 2]]

# 特别注意
# brew要求数据的形式是从0到N，不允许有跳跃
d = {yi: i for i, yi in enumerate(set(y))}
y = np.array([d[yi] for yi in y])

# 画出决定的部分
gs = gridspec.GridSpec(2, 3)
fig = plt.figure(figsize=(10, 8))

itt = itertools.product([0, 1, 2], repeat=2)

for clf, lab, grd in zip(clf_list, lbl_list, itt):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()
