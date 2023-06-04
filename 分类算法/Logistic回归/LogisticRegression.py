from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as LR

# 导入load_breast_cancer数据
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']
# 将数据划分为训练集测试集
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.2, random_state=22)
print('x_train第1行数据为：\n', x_train[0: 1], '\n', 'y_train第1个数据为：', y_train[0: 1])

# 数据标准化
stdScaler = StandardScaler()
stdScaler.fit(x_train)
x_trainStd = stdScaler.transform(x_train)
x_testStd = stdScaler.transform(x_test)
"""
- LR()对象初始化参数说明
penalty:    接收str,可选参数为“l1”和“l2”,表示惩罚的规范
tol:    接收float,表示迭代停止的容忍度，即精度要求，默认为1e-3
C:  float类型，正则化系数，默认为1.0
random_state:   随机数种子
solver: 接收str，可选参数为：newton-cg(牛顿法)、lbfgs(拟牛顿法)、liblinear(坐标轴下降法)、sag(随机平均梯度下降)、saga(线性收敛的随机优化算法)
max_iter:   模型最大迭代次数
multi_class:    接收str,为“ovr”时为二分类。为“multinomial”时为多分类
"""
# 构建LR()模型
lr_model = LR(solver='saga')
"""
- LR()对象属性及方法说明
1.属性：
coef_:  返回各特征的回归系数
2.方法.
fit(x):   代入数据集x进行训练
predict(x): 输出样本数据x的类别标签
predict_proba(x):  计算x的概率,返回的是一个N*K的二维列表，K为特征个数
score(x,y): 返回测试数据集的平均准确度，x为特征数据集，y为x对应的真实标签
decision_function(x):   样本的置信度
"""
# 训练Logistic回归模型
lr_model.fit(x_trainStd, y_train)
print('训练出来的LogisticRegression模型为：\n', lr_model)
print('各特征的相关系数为：\n', lr_model.coef_)
print('预测测试集前10个结果为：\n', lr_model.predict(x_testStd)[: 10])
print('测试集准确率为：', lr_model.score(x_testStd, y_test))
print('测试集前3个对应类别的概率为：\n', lr_model.predict_proba(x_testStd)[: 3])
print('测试集前3个对应类别的概率的log值为：\n',
      lr_model.predict_log_proba(x_testStd)[: 3])
print('测试集前3个的决策函数值为：\n',
      lr_model.decision_function(x_testStd)[: 3])
print('模型的参数为：\n', lr_model.get_params())

