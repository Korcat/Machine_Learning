from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score

X, y = make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()

y_pred = SpectralClustering().fit_predict(X)
# Calinski-Harabasz Score 14907.099436228204
print("Calinski-Harabasz Score", calinski_harabasz_score(X, y_pred))

plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


### 贝叶斯优化
# 1.构造黑盒函数
def black_box_function(n_clusters, gamma):
    y_pred = SpectralClustering(n_clusters=int(n_clusters),
                                # n_neighbors=int(n_neighbors),
                                gamma=gamma,  # float
                                ).fit_predict(X)
    res = calinski_harabasz_score(X, y_pred)  # 贝叶斯目标函数为轮廓系数,注意贝叶斯目标函数为极大型
    return res


# 2.确定域空间
param_bounds = {'n_clusters': (3, 5),
                # 'n_neighbors': (2, 15),
                'gamma': (0.01, 1),
                }

# 3.实例化对象
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=param_bounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# 4.确定迭代次数
optimizer.maximize(
    init_points=20,  # 执行随机搜索的步数
    n_iter=150,  # 执行贝叶斯优化的步数
)

# 5.搜索最优结果
print(optimizer.max)

# 结果展示
params = optimizer.max['params']
params['n_clusters'] = int(params['n_clusters'])
# params['n_neighbors'] = int(params['n_neighbors'])

y_pred_bayes = SpectralClustering(**params).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_bayes)
plt.show()
