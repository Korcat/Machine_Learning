"""
参考博客：https://blog.csdn.net/qq_52785473/article/details/126029565
"""
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
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

# 分类指标文本报告
# 这里会给出两个f1分数（其他指标也给出两个），相当于把0和1（患与不患病）看为两个种类，分别计算了这两个种类的指标值
# 最后对模型进行评价时，可以将这两个类别指标的平均值作为评价指标（如此时的平均f1分数为宏f1分数）
print('分类指标文本报告：\n', classification_report(y_true=y_test, y_pred=y_pred))


# 混淆矩阵
def plot_confusion_matrix(cm, labels_name, title="Confusion Matrix", is_norm=True, colorbar=True, cmap=plt.cm.Blues):
    if is_norm == True:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], 2)  # 横轴归一化并保留2位小数

    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 在特定的窗口上显示图像
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')  # 默认所有值均为黑色
            # plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', color="white" if i==j else "black", verticalalignment='center') # 将对角线值设为白色
    if colorbar:
        plt.colorbar()  # 创建颜色条

    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.title(title)  # 图像标题
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if is_norm == True:
        plt.savefig(r'.\cm_norm_' + '.png', format='png',dpi=200,bbox_inches='tight')
    else:
        plt.savefig(r'.\cm_' + '.png', format='png',dpi=200,bbox_inches='tight')
    plt.show()  # plt.show()在plt.savefig()之后
    plt.close()


print('混淆矩阵：\n', confusion_matrix(y_true=y_test, y_pred=y_pred))
label_name = ['no infected', 'infected']
cm = confusion_matrix(y_test, y_pred)  # 调用库函数confusion_matrix
plot_confusion_matrix(cm, label_name, "Confusion Matrix", is_norm=False)  # 调用上面编写的自定义函数
plot_confusion_matrix(cm, label_name, "Confusion Matrix", is_norm=True)  # 经过归一化的混淆矩阵

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
