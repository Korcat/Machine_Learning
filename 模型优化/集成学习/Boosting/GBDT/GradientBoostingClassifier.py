from sklearn.ensemble import GradientBoostingClassifier  # 导入GBDT分类模块（也有回归模块）
from sklearn.datasets import make_gaussian_quantiles
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
参数说明链接：
https://blog.csdn.net/qq_42554780/article/details/120563003
https://blog.csdn.net/VariableX/article/details/107200334
"""
# 解决中文显示问题
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 创建数据
# 生成2维正态分布，生成的数据按分位数分为两类，200个样本,2个样本特征，协方差系数为2
X1, y1 = make_gaussian_quantiles(cov=2, n_samples=200, n_features=2,
                                 n_classes=2, random_state=1)  # 创建符合高斯分布的数据集
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5, n_samples=300, n_features=2,
                                 n_classes=2, random_state=1)
# 将两组数据合成一组数据
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

# 构建GBDT
gbdt = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200)

gbdt.fit(X, y)

plot_step = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# meshgrid的作用：生成网格型数据
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

# 预测
# np.c_  按照列来组合数组
Z = gbdt.predict(np.c_[xx.ravel(), yy.ravel()])
# 设置维度
Z = Z.reshape(xx.shape)

plot_coloes = "br"
class_names = "AB"

plt.figure(figsize=(10, 5), facecolor="w")
# 局部子图
plt.subplot(1, 2, 1)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
for i, n, c in zip(range(2), class_names, plot_coloes):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=c, cmap=plt.cm.Paired, label=u"类别%s" % n)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc="upper right")
plt.xlabel("x")
plt.ylabel("y")
plt.title(u"GBDT分类结果,正确率为:%.2f%%" % (gbdt.score(X, y) * 100))
plt.savefig("GBDT分类结果.png")

# 获取决策函数的数值
twoclass_out = gbdt.decision_function(X)
# 获取范围
plot_range = (twoclass_out.min(), twoclass_out.max())
plt.subplot(1, 2, 2)
for i, n, c in zip(range(2), class_names, plot_coloes):
    # 直方图
    plt.hist(twoclass_out[y == i], bins=20, range=plot_range,
             facecolor=c, label=u"类别%s" % n, alpha=.5)
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc="upper right")
plt.xlabel(u"决策函数值")
plt.ylabel(u"样本数")
plt.title(u"GBDT的决策值")
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.savefig("GBDT的决策值.png")
plt.show()
