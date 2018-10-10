# coding=utf-8
from ML_actual_combat import get_train_and_test
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

get_t = get_train_and_test

# 首先是将特征(feature)和目标标签(label)分开来
# 以median_house_value作为标签，其他作为特征。
train_housing = get_t.strat_train_set.drop("median_house_value", axis=1)
train_housing_labels = get_t.strat_train_set["median_house_value"].copy()

# 下面的工作是数据清洗
"""
   从前面知道total_bedrooms存在一些缺失值，
   对于缺失值的处理有三种方案：
  1、去掉含有缺失值的个体（dropna）
  2、去掉含有缺失值的整个特征（drop）
  3、给缺失值补上一些值（0、平均数、中位数等）（fillna）
"""

# train_housing.dropna(subset=["total_bedrooms"]) # option 1
# train_housing.drop("total_bedrooms", axis=1) # option 2
# 为了得到更多的数据，选择了方案3

median = train_housing["total_bedrooms"].median()
T = train_housing["total_bedrooms"].fillna(median)  # option 3

"""
当然Scikit-Learn也存在对缺失值处理的类Imputer。
我们打算对所有地方的缺失值都补全，以防运行模型时发生错误。
使用Imputer函数需要先定义一个补缺失值的策略（如median），
由于median策略只能对实数值有效，所以需要将文本属性先去除，
然后再补缺失值。最后使用fit方法对变量执行相应操作。
"""

imputer = Imputer(strategy="median")
housing_num = train_housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

X = imputer.fit_transform(housing_num)
# print(X)


# 因为我们有一些文本属性是不可以用作median等等的操作
# 所以我们需要进行文本编码，one-hot编码，或者scikit自带的都可以


encoder = LabelEncoder()
housing_cat = train_housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)

# print(housing_cat_encoded)
# print(encoder.classes_)
# 字符编码的结果为：
# [0 0 4 ... 1 0 3]
# ['<1H OCEAN' 'INLAND' 'ISLAND' 'NEAR BAY' 'NEAR OCEAN']

# 如果是使用one-hot编码

encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
# print(housing_cat_1hot.toarray())


# 用于增加组合特征的Trainsformer
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(get_t.housing.values)

# 特征缩放
"""
    1、MinMaxScaler：将特征缩放到0-1之间，但异常值对这个影响比较大，
    比如异常值为100，缩放0-15为0-0.15;
  2、feature_range：可以自定义缩放的范围，不一定是0-1;
  3、StandardScaler：标准化（减均值，除方差），对异常值的影响较小，
    但可能会不符合某种范围
    
    需要注意：每次缩放只能针对训练集或只是测试集，
    而不能是整个数据集，这是由于测试集（或新数据）不属于训练范围。
"""


def feature_scaling():
    num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    housing_num_tr = num_pipeline.fit_transform(housing_num)

    # 由于Scikit-Learn没有处理Pandas数据的DataFrame，因此需要自己自定义一个如下：
    class DataFrameSelector(BaseEstimator, TransformerMixin):
        def __init__(self, attribute_names):
            self.attribute_names = attribute_names

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X[self.attribute_names].values

    # 由于在新版本的scikit-learn中对于fit_transform的定义不一致
    # 所以仿造LabelBinarizer()写一个只有两个输入的MyLabelBinarizer()

    class MyLabelBinarizer(TransformerMixin):
        def __init__(self, *args, **kwargs):
            self.encoder = LabelBinarizer(*args, **kwargs)

        def fit(self, x, y=0):
            self.encoder.fit(x)
            return self

        def transform(self, x, y=0):
            return self.encoder.transform(x)

    num_attribs = list(housing_num)
    cat_attribs = ["ocean_proximity"]
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', MyLabelBinarizer()),
    ])
    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    housing_prepared = full_pipeline.fit_transform(train_housing)
    # print(housing_prepared)

    return housing_prepared


train_housing_prepared = feature_scaling()
num_attribs = list(housing_num)

# feature_scaling()
