# coding=utf-8
"""
scikit-learn中提供函数GridSearchCV用于网格搜索调参，
网格搜索就是通过自己对模型需要调整的几个参数设定一些可行值，
然后Grid Search会排列组合这些参数值，每一种情况都去训练一个模型，
经过交叉验证今后输出结果。

下面为随机森林回归模型（RandomForestRegression）
的一个Grid Search的例子。

"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from ML_actual_combat.pretreatment import train_housing_prepared, train_housing_labels
import numpy as np

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(train_housing_prepared, train_housing_labels)
# 查看最优结果
# print(grid_search.best_estimator_)

# 查看交叉验证之后的最优结果，同时把方差求出，比较差距
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
