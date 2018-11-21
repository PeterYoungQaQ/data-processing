# coding=utf-8 
# @Time :2018/11/21 17:46

"""
    假设要解决一个复杂的问题，让众多学生去回答，然后汇总他们的答案。
在许多情况下，会发现这个汇总的答案比一个老师的答案要好。同样，如果
汇总了一组预测变量（例如分类器或回归因子）的预测结果，则通常会得到
比最佳个体预测变量得到更好的预测结果。这种技术被称为集成学习（Ense
mble Learning）。

"""

# 下面是一个集成Logistic回归，SVM分类，Random forest分类的投票
# 分类器，实验数据由moon产生，由于是硬投票，所以voting要设置为hard。

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

(X, y) = make_moons(1000, noise=0.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构造模型和集成模型
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()


def hard_classify():
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
        voting='hard'
    )
    voting_clf.fit(X_train, y_train)

    # 训练并预测

    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


# 如果所有分类器都能够估计分为每一类的概率，即都有predict_proba()方法，
# 那么可以对每个分类器的概率取平均，再预测具有最高类概率的类，这被称为
# 软投票(soft voting)。
# 修改SVC类使得有predict_proba()方法，并软投票
def soft_classify():
    svm_clf_2 = SVC(probability=True)
    voting_clf = VotingClassifier(
        estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf_2)],
        voting='soft'
    )
    voting_clf.fit(X_train, y_train)
    # 训练并预测
    from sklearn.metrics import accuracy_score
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, accuracy_score(y_test, y_pred))


soft_classify()