from sklearn.linear_model import Lasso
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 导入load_boston数据
housing = fetch_california_housing()
x = housing['data'][:1000]
y = housing['target'][:1000]
names = housing['feature_names']
# 将数据划分为训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                    random_state=22)
print('x_train前3行数据为：', x_train[0: 3], '\n',
      'y_train前3个数据为：', y_train[0: 3])

lasso_model = Lasso()
lasso_model.fit(x_train, y_train)

print('训练出来的Lasso模型为：\n', lasso_model)
print('scipy.sparse matrix为：\n', lasso_model.sparse_coef_)
print('预测测试集前5个结果为：\n', lasso_model.predict(x_test)[: 5])
print('测试集R2值为：', lasso_model.score(x_test, y_test))
print('测试集弹性网络路径为：\n', lasso_model.path(x_test, y_test))

# 画图
rcParams['font.sans-serif'] = 'SimHei'
fig = plt.figure(figsize=(10, 6))
y_pred = lasso_model.predict(x_test)
plt.plot(range(y_test.shape[0]), y_test, color="blue",
         linewidth=1.5, linestyle="-")
plt.plot(range(y_test.shape[0]), y_pred, color="red",
         linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
plt.show()
