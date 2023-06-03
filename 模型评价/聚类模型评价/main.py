from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载或构造数据
iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)

# 使用K-means聚类
km = KMeans(n_clusters=2, random_state=0)
km.fit(x)
# 轮廓系数

print('轮廓系数为：', silhouette_score(x, km.labels_, metric='euclidean'))

# 同质性、完整性、调和平均
km_pred = km.predict(x)
print('同质性,完整性,调和平均分别为：\n',
      homogeneity_completeness_v_measure(y, km_pred))
