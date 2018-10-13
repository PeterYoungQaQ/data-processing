# coding=utf-8

# 使用数据集MNIST
# MNIST数据集已经帮我们划分好（前60000个为训练集，后10000个位测试集）

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import cross_val_predict
import numpy as np
from matplotlib import pyplot as plt

mnist = fetch_mldata('MNIST original')
X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 对数据集中的数据进行打乱
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

# 二分类方法,训练一个SGD分类器（该分类器对大规模的数据处理较快）
# 划分数据
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# 训练模型
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 交叉验证
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# print(y_train_pred)

"""
  二分类特有的一种评价指标为查准率和查全率（Precision and Recall）以及F1指标。
Precision就是预测为正类的样本有多少比例的样本是真的正类
TP/(TP+FP)；Recall就是所有真正的正类样本有多少比例被预测为正类，
TP/（TP+FN）。
  其中TP为真正类被预测为正类，FP为负类被预测为正类，
FN为真正类被预测为负类。
  由于Precision和Recall有两个数，
如果一大一下的话不好比较两个模型的好坏，
  F1指标就是结合两者，求调和平均
F1=2∗(Precision∗Recall)/(Precision+Recall)
"""

# 求查准率和查全率以及F1
t1 = precision_score(y_train_5, y_train_pred)
t2 = recall_score(y_train_5, y_train_pred)
t3 = f1_score(y_train_5, y_train_pred)
# print(t3)

# 虽然我们没有办法改变Scikit-learn里面的predict()函数来改变分类输出
# 但我们能够通过decision_function()方法来得到输出的得分情况，
# 得分越高意味着越有把握分为这一类。
# 因此可以通过对得分设一个界（threshold），
# 得分大于threshod的分为正类，否则为负类，
# 以此来调整Precision和Recall

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


# print(precisions,recalls,thresholds)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()
