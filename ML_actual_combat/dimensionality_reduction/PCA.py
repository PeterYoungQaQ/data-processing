# coding=utf-8 
# @Time :2018/11/28 19:08
"""
降为的方法主要为两种：projection 和 Manifold Learning。
投影（Projection）：
  在大多数的真实问题，训练样例都不是均匀分散在所有的维度，许多特征都是固定的，
同时还有一些特征是强相关的。因此所有的训练样例实际上可以投影在高维空间中的低维子空间中

主成分分析（principal components analysis）是用的最出名的降维技术，
它通过确定最接近数据的超平面，然后将数据投射(project)到该超平面上。
详细内容参考：
https://blog.csdn.net/fjl_CSDN/article/details/79118212
关于SVD部分可以参考：
https://www.cnblogs.com/pinard/p/6251584.html
"""

# 产生数据
import numpy as np

x1 = np.random.normal(0, 1, 100)
x2 = x1 * 1 + np.random.rand(100)
X = np.c_[x1, x2]
# svd分解求出主成分
X_centered = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X_centered)
c1 = V.T[:, 0]
c2 = V.T[:, 1]

d = 1
Wd = V.T[:, :d]
XD = X_centered.dot(Wd)
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
XD = pca.fit_transform(X)

# 查看主成分
print(pca.components_.T)
# 显示PCA主成分比率
print("主成分方差比率为：")
print(pca.explained_variance_ratio_)
