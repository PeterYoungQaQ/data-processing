# coding=utf-8
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals import joblib

from ML_actual_combat import get_train_and_test
from ML_actual_combat.pretreatment import train_housing_prepared, train_housing_labels

get_t = get_train_and_test

# 首先尝试训练一个线性回归模型（LinearRegression）
lin_reg = LinearRegression()
lin_reg.fit(train_housing_prepared, train_housing_labels)

# 训练完成，然后评估模型，计算训练集中的均方根误差（RMSE）
housing_predictions = lin_reg.predict(train_housing_prepared)
lin_mse = mean_squared_error(train_housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)
# The answer is 68628.19819848923
# 可以看出简单的用线性回归模型的效果是很差的，偏差比较大


# 这里尝试使用决策树模型（DecisionTreeRegressor）
tree_reg = DecisionTreeRegressor()
tree_reg.fit(train_housing_prepared, train_housing_labels)
housing_predictions = tree_reg.predict(train_housing_prepared)
tree_mse = mean_squared_error(train_housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)
# 惊了！结果居然是0，
# 但是这不见的就是一个好模型，因为有可能是过拟合的结果

# 我们可以通过交叉验证的方式，对上述两个模型进行比较
tree_scores = cross_val_score(tree_reg, train_housing_prepared, train_housing_labels,
                              scoring="neg_mean_squared_error", cv=10)
lin_scores = cross_val_score(lin_reg, train_housing_prepared, train_housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
lin_rmse_scores = np.sqrt(-lin_scores)


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# display_scores(tree_rmse_scores)
# display_scores(lin_rmse_scores)

# 保存模型
# joblib.dump(lin_reg.fit(train_housing_prepared, train_housing_labels), "linear_model.pkl")

# 加载模型
# my_model_load = joblib.load("linear_model.pkl")
# print(my_model_load)