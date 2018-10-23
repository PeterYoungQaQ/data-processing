# coding=utf-8
# 最小二乘法θ^=(X**T * X) ** −1 * X**T * y
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn import datasets

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)


def linear_model():
    lin_reg = LinearRegression()
    lin_reg.fit(X, y)
    # clf.coef_和clf.intercept_就是θ
    # 其中θ是Logistic回归模型参数
    # 没有达到准确值是由于加入了随机噪声的原因
    print(lin_reg.intercept_)
    print(lin_reg.coef_)

    # 虽然能够精确的得到结果，但是这个算法的复杂度很高，
    # 计算X**T * X 的逆的算法复杂度就为O(n^2.4)−O(n^3)


# 批处理梯度下降
def batch_gradient_descent():
    eta = 0.1  # learning rate
    n_iterations = 1000
    m = 100
    theta = np.random.randn(2, 1)  # random initialization
    X_b = np.c_[np.ones((100, 1)), X]
    for iteration in range(n_iterations):
        gradients = 2 / float(m) * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradients
    print(theta)


# batch_gradient_descent()

# 随机梯度下降
def stochastic_gradient_descent():
    from sklearn.linear_model import SGDRegressor
    sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
    sgd_reg.fit(X, y.ravel())
    print(sgd_reg.intercept_)
    print(sgd_reg.coef_)


# stochastic_gradient_descent()

# 多项式模型
def polynomial_model():
    m = 100
    X = 6 * np.random.rand(m, 1) - 3
    y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)


    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(X)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    # 图形展示
    # X_poly_t = [x[0] for x in X_poly]
    # plt.scatter(X_poly_t, y, marker="o")
    # plt.show()
    print(lin_reg.intercept_)
    print(lin_reg.coef_)


# polynomial_model()

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlabel("Training set size")
    plt.ylabel("RMSE")
    plt.show()


# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)

# 考虑degree=10的模型
def degree_ten():
    polynomial_regression = Pipeline((
        ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
        ("seg_reg", LinearRegression()),
    ))
    plot_learning_curves(polynomial_regression, X, y)
    plt.ylim(0, 2)
    plt.show()


# 正则化线性模型(Regularized Linear Models)
# 为了防止过拟合，可以给模型损失函数（如均方误差）增加正则项

# Ridge Regression（L2正则化）
# 需要注意的是：在Ridge Regression之前要先对特征进行缩放（标准化或最大最小缩放），
# 这是因为Ridge Regression对特征的尺度大小很敏感

def rlm_rr_l2():
    ridge_reg = Ridge(alpha=1, solver="cholesky")
    ridge_reg.fit(X, y)
    sgd_reg = SGDRegressor(penalty="l2")
    sgd_reg.fit(X, y.ravel())
    y_2 = list(ridge_reg.predict(X))
    plt.figure()
    plt.plot(X, y_2, color='red', linewidth=4)
    plt.scatter(X, y)
    plt.show()


# rlm_rr_l2()


# Lasso Regression（L1正则化）
# 与L2正则化不同的是，正则项从二次方变为了一次放的绝对值，
# 这就带来的一个特性，不同于L2正则化使得θ在原点附近（即大部分θ都靠近0），\
# L1正则化使得θ更趋向于在坐标轴上（即大部分的θ等于零，少部分靠近零），相当于惩罚变得更大。
def rlm_rr_l1():
    # 最小二乘LASSO方法
    lasso_reg = Lasso(alpha=0.1)
    lasso_reg.fit(X, y)
    # 随机梯度下降正则化L2（线性）
    sgd_reg = SGDRegressor(penalty="l1")
    sgd_reg.fit(X, y.ravel())
    y_1 = list(lasso_reg.predict(X))
    plt.figure()
    plt.plot(X, y_1, color='red', linewidth=4)
    plt.scatter(X, y)
    plt.show()

# 需要注意的是：L1正则化后会导致在最优点附近震荡，
# 因此要像随机梯度下降一样减小学习率。
# rlm_rr_l1()


# Elastic Net是L1正则化和L2正则化的结合，通过一个参数调整比例
def elastic_net():
    elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elastic_net.fit(X, y)
    sgd_reg = SGDRegressor(penalty="elasticnet", l1_ratio=0.5)
    sgd_reg.fit(X, y.ravel())
    y_e = list(elastic_net.predict(X))
    plt.figure()
    plt.plot(X, y_e, color='red', linewidth=4)
    plt.scatter(X, y)
    plt.show()


# eastic_net()

# Logistic回归（Logistic Regression)
# Logistic回归与线性回归模型比较相似，Logistic回归在线性回归模型的基础上增加了sigmoid函数σ()
def logistic_iris():
    iris = datasets.load_iris()
    X = iris["data"][:, 3:]  # 只读取最后一个特征
    y = (iris["target"] == 2).astype(np.int)  # 取出判断是否为第三类的label

    log_reg = LogisticRegression()
    log_reg.fit(X, y)
    plt.figure()
    plt.scatter(X, y, marker="o")
    plt.show()


# logistic_iris()

# 对于Logistic回归是一个二分类器，不需要训练多个二分类器来实现多分类。
# Logistic回归可以直接扩展成一个多分类器Softmax回归。
# 与Logistic回归相似，Softmax回归计算每一类的一个概率，归为概率最大的一类
def softmax_iris():
    iris = datasets.load_iris()
    X = iris["data"][:, (2, 3)]
    y = iris["target"]
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
    softmax_reg.fit(X, y)



# softmax_iris()
