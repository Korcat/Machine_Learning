# 1. 简单堆叠3折CV分类，默认为3，可以使用cv=5改变为5折cv分类
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier

"""
stacking解释参考博客：https://blog.csdn.net/Geeksongs/article/details/120819287
"""
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
RANDOM_SEED = 42

knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(random_state=RANDOM_SEED)
gnb = GaussianNB()
lr = LogisticRegression()

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
stack_model = StackingClassifier(estimators=[('knn', knn), ('rf', rf), ('gnb', gnb)],  # 第一层分类器
                                 final_estimator=lr,
                                 # 第二层分类器，并非表示第二次stacking，而是通过logistic regression对新的训练特征数据进行训练，得到predicted label
                                 )

print('3-fold cross validation:\n')

for clf, label in zip([knn, rf, gnb, stack_model], ['KNN', 'Random Forest', 'Naive Bayes', 'StackingClassifier']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
