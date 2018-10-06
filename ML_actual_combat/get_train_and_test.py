# coding=utf-8

from ML_actual_combat import read_data
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

"""
    这里是使用的是直接去分开测试集和数据集的方法，但是会导致每次数据集
和测试集的内容都不一样，就没有办法直接的将这两个分开，所以我们直接用
scikit-learn提供的方法来做
    但是其这样做的效率会变得更慢就是了
"""
# import numpy as np

# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(read_data.housing, 0.2)

housing = read_data.housing

train_set, test_set = train_test_split(housing,
                                       test_size=0.2, random_state=42)
# print(len(train_set), "train +", len(test_set), "test")

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)


def show_data():
    housing["income_cat"].hist(bins=50, figsize=(20, 15))
    plt.show()


# 分层抽样的方法

global strat_train_set, strat_test_set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
