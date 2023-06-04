import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
# 导入iris数据
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
X = X[:, :2]  # 只取前两列

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)  # 划分数据，random_state固定划分方式
# 导入模型


# 训练模型
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
# 查看各项得分
print("y_pred", y_pred)
print("y_test", y_test)
print("score on train set", knn.score(X_train, y_train))
print("score on test set", knn.score(X_test, y_test))
print("accuracy score", accuracy_score(y_test, y_pred))


def plot_decision_boundary(clf, axes):
    xp = np.linspace(axes[0], axes[1], 300)  # 均匀300个横坐标
    yp = np.linspace(axes[2], axes[3], 300)  # 均匀300个纵坐标
    x1, y1 = np.meshgrid(xp, yp)  # 生成300x300个点
    xy = np.c_[x1.ravel(), y1.ravel()]  # 按行拼接，规范成坐标点的格式
    y_pred = clf.predict(xy).reshape(x1.shape)  # 训练之后平铺
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])
    plt.contourf(x1, y1, y_pred, alpha=0.3, cmap=custom_cmap)


plot_decision_boundary(knn, axes=[4, 8, 1.5, 5])
# 画三种类型的点
p1 = plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue')
p2 = plt.scatter(X[y == 1, 0], X[y == 1, 1], color='green')
p3 = plt.scatter(X[y == 2, 0], X[y == 2, 1], color='red')
# 设置注释
plt.legend([p1, p2, p3], iris['target_names'], loc='upper right')
plt.show()
