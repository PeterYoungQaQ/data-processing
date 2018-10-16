# coding=utf-8

# 使用数据集MNIST
# MNIST数据集已经帮我们划分好（前60000个为训练集，后10000个位测试集）

from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import numpy.random as rnd
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
sgd_clf = SGDClassifier(random_state=42, max_iter=None, tol=None)
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

def plot_precision_recall_vs_threshold(precision, recall, threshold):
    plt.plot(threshold, precision[:-1], "b--", label="Precision")
    plt.plot(threshold, recall[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

"""
将二分类器扩展到多分类器一般有两种做法。
  1、OVA(one-versus-all)：比如分类数字（0-9），
    则训练10个分类器（是否为0的分类器，是否为1的分类器.，…，是否为9的分类器），
    每一个分类器最后会算出一个得分，判定为最高分的那一类
  
    2、OVO(one-versus-one)：每个类之间训练一个分类器（比如0和1训练一个分类器，1-3训练一个分类器），
    这样总共有N*(N-1)/2个分类器，哪个类得分最高判定为那一类。
"""


# 下面是一个二分类器SGD分类器扩展为多分类器用作数字分类的例子(使用的是OVA的方法）
def multiple_claasifier():
    sgd_clfs = SGDClassifier(random_state=42, max_iter=None, tol=None)
    sgd_clfs.fit(X_train, y_train)
    some_digit = X[1]
    sgd_clfs.predict([some_digit])
    some_digit_scores = sgd_clfs.decision_function([some_digit])
    print(some_digit_scores)


# multiple_claasifier()


"""
        对于分类任务，同样也可以采取交叉验证法，
不同的是，误差不是均方误差，而是准确率（或者交叉熵）。
和之前交叉验证的函数相同为cross_val_score，不过scoring为accuracy。
"""


def classifier_effect():
    # K折交叉验证（k-fold）
    # cv = k
    # 把初始训练样本分成k份，其中（k-1）份被用作训练集，
    # 剩下一份被用作评估集，这样一共可以对分类器做k次训练，
    # 并且得到k个训练结果。
    result = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
    # print(result)
    # 当然也可以采用预处理（标准化）来增加准确率
    scaler = StandardScaler()
    X_train_scale = scaler.fit_transform(X_train.astype(np.float64))
    new_result = cross_val_score(sgd_clf, X_train_scale, y_train, cv=3, scoring="accuracy")
    # print(new_result)
    return X_train_scale


X_train_scaled = classifier_effect()

# 接下来要进行简单的错误分析
"""
    首先要使用cross_val_predict计算交叉验证后验证集部分
的预测分类结果（而不是正确率），然后根据真正的标签和预测结果对比，画出混淆矩阵，
第 i 行第 j 列的数字代表数字 i 被预测为数字 j 的个数总和,对角线上即为正确分类的次数
"""


def error_analysis():
    y_train_pre = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
    conf_mx = confusion_matrix(y_train, y_train_pre)
    print(conf_mx)
    plt.matshow(conf_mx, cmap=plt.cm.gray)
    plt.show()
    # 只看错误的分布
    row_sums = conf_mx.sum(axis=1, keepdims=True)
    norm_conf_mx = conf_mx / row_sums.astype(np.float64)
    # fill_diagonal(, 0)将对角元素设置为0，不考虑正确分组
    np.fill_diagonal(norm_conf_mx, 0)
    plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
    plt.show()


# error_analysis()


# 多标签分类
# 例子中的任务是：分类数据 是否大于等于7 以及 是否为奇数 这2个标签
# 使用的是KNN算法
def multilabel_classification():
    some_digit = X[1]
    y_train_large = (y_train >= 7)
    y_train_odd = (y_train % 2 == 1)
    # numpy.c_()
    # Translates slice objects to concatenation along the second axis.
    y_multilabel = np.c_[y_train_large, y_train_odd]
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_multilabel)
    result = knn_clf.predict([some_digit])
    print(result)


# multilabel_classification()

# 多输出分类

def multioutput_classification():
    # 生成噪声图
    noise1 = rnd.randint(0, 100, (len(X_train), 784))
    noise2 = rnd.randint(0, 100, (len(X_test), 784))
    X_train_mod = X_train + noise1
    X_test_mod = X_test + noise2
    y_train_mod = X_train
    y_test_mod = X_test
    plt.subplot(1, 2, 1)
    plt.imshow(X_train_mod[36000].reshape(28, 28), cmap=plt.cm.gray)
    plt.subplot(1, 2, 2)
    plt.imshow(X_train[36000].reshape(28, 28), cmap=plt.cm.gray)
    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train_mod, y_train_mod)
    clean_digit = knn_clf.predict([X_train_mod[36000]])
    plt.imshow(clean_digit.reshape(28, 28), cmap=plt.cm.gray)
    plt.show()


# multioutput_classification()
