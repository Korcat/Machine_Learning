import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# 导入数据
iris = datasets.load_iris()
x = iris.data
y = iris.target
# 绘制样本数据
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.title('iris', size=17)
plt.show()

"""
GaussianMixture()对象初始化参数说明：
n_components:   接收int,聚类的簇个数
tol:   EM算法迭代的停止阈值，默认为0.0001
max_iter:   最大迭代次数
init_params:    接收str,表示初始化簇中心点的方法，有kmeans,random,默认为kmeans
"""
# 构建聚类数为3的GMM模型
gmm = GaussianMixture(n_components=3)
gmm.fit(x)
print('GMM模型：\n', gmm)

"""
模型属性参数：
weights_:   各个高斯模型的权重
means_: 各个高斯模型的均值
covariances_:   各个高斯模型的协方差
"""
print('GMM模型的权重为：', gmm.weights_)
print('GMM模型的均值为：\n', gmm.means_)
print('GMM模型的均值为：\n', gmm.covariances_)

"""
模型方法：
aic(x): 生成训练数据x的Akaike信息准则
bic(x): 生成训练数据x的贝叶斯信息准则
fit(x): 训练数据
fit_predict(x): 训练数据并返回样本标签
predict(x): 使用训练过的数据获取样本标签
predict_proba(x):   返回数据x的每个高斯模型的后验概率
"""
print(f'GMM模型的aic为：{gmm.aic(x)}')
print(f'GMM模型的bic为：{gmm.bic(x)}')
print(f'样本的每个高斯模型的后验概率为：{gmm.predict_proba(x)}')

# 获取GMM模型聚类结果
gmm_pre = gmm.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=gmm_pre)
plt.title('GMM', size=17)
plt.show()

# K-means聚类
km_pre = KMeans(n_clusters=3).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=km_pre)
plt.title('K-means', size=17)
plt.show()
