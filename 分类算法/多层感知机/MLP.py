from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_wine  # 导入红酒数据集
from sklearn.model_selection import train_test_split  # 划分数据集的模块
from sklearn.preprocessing import StandardScaler

wine = load_wine()
x = wine.data
y = wine.target
# 需要对数据进行标准化处理
std_transfer = StandardScaler()
x_std = std_transfer.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=22)  # 测试用
"""
MLPClassifier()对象初始化参数、属性及方法说明：https://blog.csdn.net/weixin_44491423/article/details/116711606
"""
mlp_model = MLPClassifier(max_iter=1000, random_state=3)
mlp_model.fit(x_train, y_train)

print('训练出来的多层感知机模型为：\n', mlp_model)
print('分类模型的类别标签为：', mlp_model.classes_)
print('当前损失值为：', mlp_model.loss_)
print('迭代次数为：', mlp_model.n_iter_)
print('神经网络层数为：', mlp_model.n_layers_)
print('输出个数为：', mlp_model.n_outputs_)
print('输出激活函数的名称为：', mlp_model.out_activation_)
print('权重矩阵为：\n', mlp_model.coefs_)
print('偏差向量为：\n', mlp_model.intercepts_)

print('预测测试集前10个结果为：\n', mlp_model.predict(x_test)[: 10])
print('测试集准确率为：', mlp_model.score(x_test, y_test))
