from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.cluster.hierarchy import linkage, dendrogram

# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target

# 绘制树状图
plt.figure(figsize=(20,6))
Z = linkage(x, method='ward', metric='euclidean')
p = dendrogram(Z, 0)
plt.show()


# 单链接层次聚类
"""
AgglomerativeClustering()初始化参数说明
n_clusters: 接收int,表示聚类簇的数量
affinity:   接收str,表示计算距离的方法，euclidean为欧式距离，manhattan为曼哈顿距离，cosine余弦距离
linkage:    接收str,链接算法， ward：组间距离等于两类对象之间的最小距离。（即single-linkage聚类）
                          average：组间距离等于两组对象之间的平均距离。（average-linkage聚类）
                          complete：组间距离等于两组对象之间的最大距离。（complete-linkage聚类）
"""
clusing = AgglomerativeClustering(n_clusters=3)
clusing.fit(x)
print('单链接层次聚类模型为：\n', clusing)
print('簇类别标签为：\n', clusing.labels_)
print('叶节点数量为：', clusing.n_leaves_)

# 绘制单链接聚类结果
cw_ypre = AgglomerativeClustering(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=cw_ypre)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.title('单链接聚类', size=17)
plt.show()

# 绘制均链接聚类结果
cw_ypre = AgglomerativeClustering(linkage='average',
                                  n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=cw_ypre)
plt.title('均链接聚类', size=17)
plt.show()

# 绘制全链接聚类结果
cw_ypre = AgglomerativeClustering(linkage='complete', n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=cw_ypre)
plt.title('全链接聚类', size=17)
plt.show()
