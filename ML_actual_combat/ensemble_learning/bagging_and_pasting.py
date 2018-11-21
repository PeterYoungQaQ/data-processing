# coding=utf-8 
# @Time :2018/11/21 19:52
"""
    对每个分类器使用相同的算法，但是要在训练集的不同随机子集上进行
训练。如果抽样时有放回，称为Bagging；当抽样没有放回，称为Pasting。
具体教程参考https://blog.csdn.net/fjl_CSDN/article/details/79038622
"""

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

(X, y) = make_moons(1000, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
    下面使用bagging算法训练模型，选择决策树分类器作为训练算法；
n_estimators表示产生分类器的数目；max_samples为每个分类器分得
的样本数；bootstrap=True表示使用bagging算法，否则为pasting算
法；n_jobs表示使用CPU核的数目，-1代表把能用的都用上。
"""

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1
)

pas_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    max_samples=100, bootstrap=False, n_jobs=-1
)

bag_clf.fit(X_train, y_train)
y_pred_b = bag_clf.predict(X_test)
pas_clf.fit(X_train, y_train)
y_pred_p = pas_clf.predict(X_test)

print(bag_clf.__class__.__name__, accuracy_score(y_test, y_pred_b))
print(pas_clf.__class__.__name__, accuracy_score(y_test, y_pred_p))

"""
    由于bagging算法采用有放回的抽样方式（自助采样法），
假设训练集有m个样本，每次抽取一个后放回，直到抽到m个样本，
那么样本始终没有被抽到的概率为(1−1m)m(1−1m)m，取极限得
lim m→∞ (1−1/m)**m=1/e≈0.368
    这意味对于每一个分类器大约有36.8%的样本没有用于训练，
这样的样本成为Out-of-Bag，所以可以使用这些样本得到结果的
平均值来用于验证模型，而不需要划分训练验证集或交叉验证。
    在Scikit-learn中只需要设置参数oob_score=True即可使用
这种方法估计
"""
bag_clf_o = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True)
bag_clf_o.fit(X_train, y_train)
print("oob", bag_clf_o.oob_score_)

y_pred_o = bag_clf_o.predict(X_test)
print("test", accuracy_score(y_test, y_pred_o))
