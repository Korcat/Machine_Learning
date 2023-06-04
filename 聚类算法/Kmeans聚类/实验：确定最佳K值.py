from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings

warnings.simplefilter('ignore')

# 导入数据
wine = datasets.load_wine()
x = wine.data
y = wine.target

kmeans_per_k = [KMeans(n_clusters=k, n_init=10).fit(x) for k in range(2, 10)]
inertias = [model.inertia_ for model in kmeans_per_k]
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(2, 10), inertias, 'bo-')
plt.title("不同K值的inertia值")
plt.show()

silhouette_scores = [silhouette_score(x, model.labels_, metric='euclidean') for model in kmeans_per_k]
print(silhouette_scores)
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.plot(range(2, 10), silhouette_scores, 'bo-')
plt.title("不同K值的轮廓系数")
plt.show()
