# 对svm模型进行交叉验证
from sklearn import svm
from sklearn.model_selection import cross_val_score  # 导入交叉验证模块
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
clf = svm.SVC(kernel='linear', C=1)
"""
cv: 接收int，表示几折交叉验证
"""
score = cross_val_score(clf, cancer.data, cancer.target, cv=5)  # 5折交叉验证
print('交叉验证结果为：\n', score)
