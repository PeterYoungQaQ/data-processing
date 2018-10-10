# coding=utf-8
"""
    由于之前的网格搜索搜索空间太大，
而机器计算能力不足，则可以通过给参数设定一定的范围，
在范围内使用随机搜索选择参数，
    随机搜索的好处是能在更大的范围内进行搜索，
并且可以通过设定迭代次数n_iter，
根据机器的计算能力来确定参数组合的个数
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from ML_actual_combat.pretreatment \
    import train_housing_labels, train_housing_prepared, encoder, num_attribs
import numpy as np

param_ran = {'n_estimators': range(30, 50), 'max_features': range(3, 8)}
forest_reg = RandomForestRegressor()
random_search = RandomizedSearchCV(forest_reg, param_ran,
                                   cv=5, scoring='neg_mean_squared_error')
random_search.fit(train_housing_prepared, train_housing_labels)
# print(random_search.best_estimator_)

# 查看交叉验证之后的最优结果，同时把方差求出，比较差距
# cvres = random_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

"""
    假设现在调参以后得到最好的参数模型，
然后可以查看每个特征对预测结果的贡献程度，
根据贡献程度，可以删减减少一些不必要的特征。
"""


def feature_importance():
    feature_importances = random_search.best_estimator_.feature_importances_
    extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
    cat_one_hot_attribs = list(encoder.classes_)
    attributes = num_attribs + extra_attribs + cat_one_hot_attribs
    sorted(zip(feature_importances, attributes), reverse=True)


feature_importance()