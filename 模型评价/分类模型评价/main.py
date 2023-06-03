"""
参考博客：https://blog.csdn.net/qq_52785473/article/details/126029565
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 加载或构造数据
cancer = load_breast_cancer()
x = cancer.data
y = cancer.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)

SVC_model = SVC()
SVC_model.fit(x_train, y_train)
y_pred = SVC_model.predict(x_test)

# 准确率
print('准确率为：', accuracy_score(y_true=y_test, y_pred=y_pred))

# 混淆矩阵
print('混淆矩阵为：\n', classification_report(y_true=y_test, y_pred=y_pred))

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, label='ROC(area = %0.2f)' % (roc_auc))
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel("FPR (假正率)")
plt.ylabel("TPR (真正率)")
plt.title("ROC曲线, ROC(AUC = %0.2f)" % (roc_auc))
plt.show()
