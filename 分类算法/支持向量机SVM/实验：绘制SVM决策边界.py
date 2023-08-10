"""
可以用于多分类可视化
"""
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

iris = datasets.load_iris()
X = iris['data'][:, [2, 3]]
y = iris['target']
X = pd.DataFrame(X)
y = pd.DataFrame(y)
data = pd.merge(X, y, left_index=True, right_index=True, how='outer')
data.columns = ['x1', 'x2', 'y']

h = 0.002
x_min, x_max = data.x1.min() - 0.2, data.x1.max() + 0.2
y_min, y_max = data.x2.min() - 0.2, data.x2.max() + 0.2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
sns.scatterplot(x=data.x1, y=data.x2, hue=data.y)
plt.show()
X = data[['x1', 'x2']]
y = data.y

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)  # 80%和20%划分X和y
clf = SVC(C=0.1, kernel='linear')
clf.fit(X_train, y_train)
y_pre = clf.predict(X)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
sns.scatterplot(x=data.x1, y=data.x2, hue=y_pre)
plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.2)
plt.show()

print(accuracy_score(data.y, y_pre))
