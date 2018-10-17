# coding=utf-8
# 最小二乘法θ^=(X**T * X) ** −1 * X**T * y
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

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

    from sklearn.preprocessing import PolynomialFeatures
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

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
