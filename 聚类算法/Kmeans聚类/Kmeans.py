from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 构建并训练K-means模型
"""
KMeans对象初始化参数说明
n_clusters: 接收int,表示分类簇的个数
init:       接收str,表示初始簇中心的获取方法，默认为Kmeans++
max_iter:   最大迭代次数，默认为30
tol:    接收float,表示容忍度，计算法收敛条件
"""
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(x)
print('K-means模型为：\n', kmeans)
print('簇的质心为：\n', kmeans.cluster_centers_)
print('样本所属的簇为：\n', kmeans.labels_)
print('样本到类中心的距离之和为：', kmeans.inertia_)
# 获取模型聚类结果
y_pre = kmeans.predict(x)
# 绘制iris原本的类别
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

# 绘制kmeans聚类结果
plt.scatter(x[:, 0], x[:, 1], c=y_pre)
plt.show()
