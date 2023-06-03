import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

# 生成两簇非凸数据
# 团状数据
x1, y2 = datasets.make_blobs(n_samples=1000, n_features=2,
                             centers=[[1.2, 1.2]], cluster_std=[[.1]],
                             random_state=9)
# 圆圈数据
x2, y1 = datasets.make_circles(n_samples=5000, factor=.6, noise=.05)
x = np.concatenate((x1, x2))
plt.scatter(x[:, 0], x[:, 1], marker='o')
plt.show()

# 生成DBSCAN模型
"""
DBSCAN()对象初始化参数
eps:    接收float,表示邻域的半径
min_samples: 接收int，当一个点邻域内的点大于这个数值时，该点就被认定为核心点
"""
dbs = DBSCAN()
dbs.fit(x)
print('DBSCAN模型:\n', dbs)
print('DBSCAN模型的簇标签为：', dbs.labels_) # 如果是噪声点，其label为-1
print('核心样本的位置为：', dbs.core_sample_indices_)
# 调整eps参数和min_samples参数
ds_pre = DBSCAN(eps=0.1, min_samples=12).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ds_pre)
plt.title('DBSCAN', size=17)
plt.show()

# K-means聚类
km_pre = KMeans(n_clusters=3, random_state=9).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=km_pre)
plt.title('K-means', size=17)
plt.show()

