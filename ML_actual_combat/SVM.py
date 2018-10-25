# coding=utf-8
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures

iris = datasets.load_iris()
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = (iris["target"] == 2).astype(np.float64)  # Iris-Virginica
svm_clf = Pipeline((
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC(C=1, loss="hinge")),
))
svm_clf.fit(X, y)

# 预测
svm_clf.predict([[5.5, 1.7]])

X, y = make_moons(n_samples=100, noise=0.15, random_state=42)


def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "bs")
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "g^")
    plt.axis(axes)
    plt.grid(True, which="both")
    plt.xlabel(r"$x_l$")
    plt.ylabel(r"$x_2$")


# contour函数是画出轮廓，需要给出X和Y的网格，以及对应的Z，它会画出Z的边界（相当于边缘检测及可视化）
def plot_predict(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contour(x0, x1, y_pred, cmap=plt.cm.winter, alpha=0.5)
    plt.contour(x0, x1, y_decision, cmap=plt.cm.winter, alpha=0.2)


polynomial_svm_clf = Pipeline([("poly_featutres", PolynomialFeatures(degree=3)),
                               ("scaler", StandardScaler()),
                               ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
                               ])
polynomial_svm_clf.fit(X, y)
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plot_predict(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])


# plt.show()

def ball_data():
    # 构造球型数据集
    (X, y) = make_moons(200, noise=0.2)
    # 使用SVC类中的多项式核训练
    poly_kernel_svm_clf = Pipeline((
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
    ))
    poly_kernel_svm_clf.fit(X, y)

    xx, yy = np.meshgrid(np.arange(-2, 3, 0.01), np.arange(-1, 2, 0.01))
    y_new = polynomial_svm_clf.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.contourf(xx, yy, y_new.reshape(xx.shape), cmap="PuBu")
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=y)
    plt.show()


ball_data()