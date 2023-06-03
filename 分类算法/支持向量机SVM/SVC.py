from sklearn.datasets import load_wine  # 导入红酒数据集
from sklearn.model_selection import train_test_split  # 划分数据集的模块
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# 导入数据集
wine = load_wine()
x = wine.data
y = wine.target
# 数据预处理（标准化）
std_transfer = StandardScaler()  # 生成标准化模型对象
x_std = std_transfer.fit_transform(x)
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(x_std, y,
                                                    test_size=0.2, random_state=22)
"""
- SVC()对象初始化参数说明
kernel: 接收str，可选参数为：linear（线性）、poly（多项式）、rbf（高斯核）、sigmoid、precomputed或者一个自定义的核函数，默认为rbf
gamma:  接收float,表示核函数为rbf、poly和sigmoid时的核函数系数，它是特征维数的倒数，默认为auto
tol:    接收float,表示迭代停止的容忍度，即精度要求,默认为1e-3
C:  float类型，正则化系数，默认为1.0
random_state:   随机数种子
max_iter:   模型最大迭代次数
"""
svc = SVC()
"""
- SVC()对象属性和方法说明
1.属性
n_support_:各类的支持向量的个数
support_：各类的支持向量在训练样本中的索引
2.方法
predict：返回一个数组表示个测试样本的类别。
predict_prob：返回一个数组表示测试样本属于每种类型的概率。
decision_function：返回一个数组表示测试样本到对应类型的超平面距离。
score：获取预测结果准确率。
"""
svc.fit(x_train, y_train)
print('前5个支持向量的索引为：', svc.support_[0: 5])
print('第1个支持向量为：\n', svc.support_vectors_[0: 1])
print('每个类别支持向量的个数为：', svc.n_support_)
print('支持向量的系数为：\n', svc.dual_coef_)
print('模型的截距值为：', svc.intercept_)
print('预测测试集前10个结果为：\n', svc.predict(x_test)[: 10])
print('测试集准确率为：', svc.score(x_test, y_test))
print('测试集前10个距超平面的距离为：\n', svc.decision_function(x_test)[: 10])
