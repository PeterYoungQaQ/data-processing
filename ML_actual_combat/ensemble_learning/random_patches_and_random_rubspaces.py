# coding=utf-8 
# @Time :2018/11/22 14:14
"""
    除了能随机选样本创建多个子分类器以外还能够随机选择特征来创建多个
子分类器，通过参数max_features和bootstrap_features实现，其含义
与max_samples和bootstrap类似。对特征进行采样能够提升模型的多样性，
增加偏差，减少方差。
    当处理高维(多特征)数据（例如图像）时，这种方法比较有用。同时对
训练数据和特征进行抽样称为Random Patches，只针对特征抽样而不针对
训练数据抽样是Random Subspaces。
    具体内容参考
https://blog.csdn.net/fjl_CSDN/article/details/79038622
"""
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import matplotlib
from sklearn.datasets import fetch_mldata

(X, y) = make_moons(1000, noise=0.5)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

"""
    随机森林算法是以决策树算法为基础，通过bagging算法采样训练样本，
再抽样特征，3者组合成的算法。
"""
rnd_clf = RandomForestClassifier(n_estimators=500,
                                 max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)

print(accuracy_score(y_test, y_pred_rf))

# 特征重要性(Feature Importance)
"""
    由于决策树算法根据最优特征分层划分的，即根部的特征更为重要，
而底部的特征不重要（不出现的特征更不重要）。根据这个可以判断特征
的重要程度，
"""


def feature_importance():
    iris = load_iris()
    rnd_clf_f = RandomForestClassifier(n_estimators=500,
                                       n_jobs=-1)
    rnd_clf_f.fit(iris["data"], iris["target"])
    for name, score in zip(iris["feature_names"],
                           rnd_clf_f.feature_importances_):
        print(name, score)


# feature_importance()
# 随机森林还能对图像中像素（特征）的重要程度，下面以MNIST图像为例子。

def rp_photo():
    mnist = fetch_mldata('MNIST original')
    rnd_clf_p = RandomForestClassifier(random_state=42, n_estimators=100)
    rnd_clf_p.fit(mnist["data"], mnist["target"])

    importance = rnd_clf_p.feature_importances_.reshape(28, 28)
    plt.imshow(importance, cmap=matplotlib.cm.hot)
    plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
    plt.show()


# rp_photo()
