from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
x = cancer['data']
y = cancer['target']
print('cancer数据集维数为：', x.shape, '\n', 'cancer样本个数为：', y.shape)

# 按7:3的比例划分数据集
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0)
print('训练集维数为：', x_train.shape, '\n', '训练集样本个数为：', y_train.shape)
print('测试集维数为：', x_test.shape, '\n', '测试集样本个数为：', y_test.shape)
