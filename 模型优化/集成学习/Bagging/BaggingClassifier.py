# 产生样本数据集
from sklearn.model_selection import cross_val_score
from sklearn import datasets
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
X, y = iris.data, iris.target

print('==================Bagging 元估计器============')

bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5, oob_score=True)
scores = cross_val_score(bagging, X, y)
bagging.fit(X, y)
print('Bagging准确率：', scores.mean())
print(bagging.estimators_features_)
print(bagging.oob_score_)
