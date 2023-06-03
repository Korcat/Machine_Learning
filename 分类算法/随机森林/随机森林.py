from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine  # 导入红酒数据集
from sklearn.model_selection import train_test_split  # 划分数据集的模块

wine = load_wine()
x = wine.data
y = wine.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=22)  # 测试用
"""
参数解释：https://blog.csdn.net/qq_42479987/article/details/109549166
"""
rf_model = RandomForestClassifier()
"""
RandomForestClassifier()对象属性及方法
https://blog.csdn.net/VariableX/article/details/107190275
"""
rf_model.fit(x_train, y_train)
print('训练出来的随机森林模型为：\n', rf_model)
print('训练出来的前2个决策树模型为：\n', rf_model.estimators_[0: 2])
print('预测测试集前10个结果为：\n', rf_model.predict(x_test)[: 10])
print('测试集准确率为：', rf_model.score(x_test, y_test))
