from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine  # 导入红酒数据集
from sklearn.model_selection import train_test_split  # 划分数据集的模块

wine = load_wine()
x = wine.data
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)  # 测试用
"""
- GaussianNB()对象初始化参数
priors: 接收array,表示先验概率大小，默认为None
"""
gnb_model = GaussianNB()
"""
- GaussianNB()对象属性和方法
1.属性
class_prior_:    每个类别的概率
class_count_:   每个类别的样本数量
2.方法
fit(x):   代入数据集x进行训练
partial_fit(x,y):   追加训练数据，即当数据量非常大的时候可以将其划分为多个数据集进行追加训练
predict(x): 输出样本数据x的类别标签
predict_proba(x):  计算x的概率,返回的是一个N*K的二维列表，K为特征个数
score(x,y): 返回测试数据集的平均准确度，x为特征数据集，y为x对应的真实标签
decision_function(x):   样本的置信度
"""
gnb_model.fit(x_train, y_train)

print('训练出来的朴素贝叶斯模型为：', gnb_model)
print('每个类别出现的概率为：', gnb_model.class_prior_)
print('每个类别训练样本的数量为：', gnb_model.class_count_)
print('每个类别中每个特征的均值为：\n', gnb_model.theta_)
print('每个类别中每个特征的方差为：\n', gnb_model.var_)

print('预测测试集前10个结果为：\n', gnb_model.predict(x_test)[: 10])
print('测试集准确率为：', gnb_model.score(x_test, y_test))
print('追加训练数据后的模型为：', gnb_model.partial_fit(x_test, y_test))
