"""
不能用于二分类以上的数据可视化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

# 创建数据点并分类
a, b = make_blobs(n_samples=100, centers=2, random_state=6)

# 以散点图的形式把数据画出来

plt.scatter(a[:, 0], a[:, 1], c=b, s=30, cmap=plt.cm.Paired)
plt.xlabel('x')
plt.ylabel('y')

# 创建一个多项式内核的支持向量机模型
clf = svm.SVC(kernel='poly', C=1000)
clf.fit(a, b)

# 建立图像坐标
axis = plt.gca()
xlim = axis.get_xlim()
ylim = axis.get_ylim()

# 生成两个等差数列
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
# print("xx:", xx)
# print("yy:", yy)

X, Y = np.meshgrid(xx, yy)
# print("X:", X)
# print("Y:", Y)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = clf.decision_function(xy)
Z = Z.reshape(X.shape)
# 画出分界线
axis.contour(X, Y, Z, colors='purple', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
axis.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=30, linewidths=1,
             facecolors='r')  # 画出支持向量点（在决策边界上的样本点）
plt.show()
