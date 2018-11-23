# coding=utf-8 
# @Time :2018/11/23 14:18

"""
    Boosting是将弱学习器集成为强学习器的方法，主要思想是按顺序训练学
习器，以尝试修改之前的学习器。Boosting的方法有许多，最为有名的方法为
AdaBoost（Adaptive Boosting）和Gradient Boosting。
具体内容参考:
https://blog.csdn.net/fjl_CSDN/article/details/79038622
"""
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy.random as rnd
import numpy as np

(X, y) = make_moons(1000, noise=0.5)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=42)

# 一个新的学习器会更关注之前学习器分类错误的训练样本。因此新的学习器
# 会越来越多地关注困难的例子。这种技术成为AdaBoost。

"""
下面介绍AdaBoost算法是怎么工作的：
  (1)假设有m个训练样本，初始化每个样本的权值w(i)w(i)为1m1m，
        经过第j个学习器训练以后，对训练样本计算加权错误率rj:
        
rj=∑(i=1->m || yj**(i)≠y**(i)) w(i) / ∑(i=1->m) w(i)
  
(2)然后计算每个学习器对应的权值αj。当rj比较小时，说明该
学习器的准确率越高，比随机猜（0.5）越好，分配的权值也越大；
如果随机猜（0.5），权值为0；如果小于随机猜，则为负值。其中
η为学习率（和梯度下降法有点相似） 
     
     αj=ηlog((1−rj) / rj)
     
（3）更新样本权值w(i)，将没有预测出来的样本权值变大，以便后续
的学习器重点训练。当然这个计算完以后需要归一化。 
（4）重复（1）（2）（3）步骤不断更新权值和训练新的学习器，直到
学习器到一定的数目。
（5）最终得到N个学习器，计算每个学习器对样本的加权和，并预测为
加权后最大的一类。
"""


# 可以看到上述的AdaBoost是二分类学习器，Scikit-learn中
# 对应为AdaBoostClassifier类，如果要多分类，则可以设置
# 参数algorithm=”SAMME”,如果需要predict_proba()方法，
# 则设置参数algorithm=”SAMME.R”

def booting_test():
    ada_clf = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=1), n_estimators=200,
        algorithm="SAMME.R", learning_rate=0.5
    )
    ada_clf.fit(X_train, y_train)
    y_pred_ad = ada_clf.predict(X_test)

    print(accuracy_score(y_test, y_pred_ad))


# booting_test()

"""
    和AdaBoost类似，Gradient Boosting也是逐个训练学习器，
尝试纠正前面学习器的错误。不同的是，AdaBoost纠正错误的方法是
更加关注前面学习器分错的样本，Gradient Boosting（适合回归
任务）纠正错误的方法是拟合前面学习器的残差（预测值减真实值）。

tips:相当于就是对前一个学习器的残差图像点去做一个拟合效果，目的是
找到那些残差比较多的地方的修改方式，带回原来的拟合图像里边，可以减
小残差的数值
"""


# 训练以决策树回归为基算法
# 根据上一个学习器的残差训练3个学习器。数据使用二次加噪声数据。

def gradient_b_test():
    # 准备数据
    X_g = rnd.rand(200, 1) - 1
    y_g = 3 * X_g ** 2 + 0.05 * rnd.randn(200, 1)
    # 训练第一个模型
    tree_reg1 = DecisionTreeRegressor(max_depth=2)
    tree_reg1.fit(X_g, y_g)
    # 根据上一个模型的残差训练第二个模型
    y2 = y_g - tree_reg1.predict(X_g)
    tree_reg2 = DecisionTreeRegressor(max_depth=2)
    tree_reg2.fit(X_g, y2)
    # 再根据上一个模型的残差训练第三个模型
    y3 = y2 - tree_reg2.predict(X_g)
    tree_reg3 = DecisionTreeRegressor(max_depth=2)
    tree_reg3.fit(X_g, y3)
    # 预测
    X_new_ = np.array([0.5])
    X_new = X_new_.reshape(-1, 1)
    y_pred_gb = sum(tree.predict(X_new) for tree in (tree_reg1, tree_reg2, tree_reg3))
    print(y_pred_gb)

    # 为了找到最优学习器的数量，可以使用early stopping方法
    # 对应可以使用staged_predict()方法，该方法能够返回
    # 每增加一个学习器的预测结果。

    X_train_g, X_val_g, y_train_g, y_val_g = train_test_split(X_g, y_g)
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    gbrt.fit(X_train_g, y_train_g)
    errors = [mean_squared_error(y_val_g, y_pred_g) for y_pred_g in gbrt.staged_predict(X_val_g)]
    bst_n_estimators = np.argmin(errors)
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
    gbrt_best.fit(X_train_g, y_train_g)


# gradient_b_test()
