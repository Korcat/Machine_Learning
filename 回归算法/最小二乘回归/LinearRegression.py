from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.linear_model import LinearRegression

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

# 使用LinearRegression类构建线性回归模型

lr_model = LinearRegression()
# 训练模型
lr_model.fit(x_train, y_train)
print('训练出来的LinearRegression模型为：\n', lr_model)
print('LinearRegression模型中各特征系数为：\n', lr_model.coef_)
print('LinearRegression模型中截距为：', lr_model.intercept_)

print('预测测试集前5个结果为：\n', lr_model.predict(x_test)[: 5])
print('测试集R2值为：', lr_model.score(x_test, y_test))

# 画图
rcParams['font.sans-serif'] = 'SimHei'
fig = plt.figure(figsize=(10, 6))
y_pred = lr_model.predict(x_test)
plt.plot(range(y_test.shape[0]), y_test, color="blue",
         linewidth=1.5, linestyle="-")
plt.plot(range(y_test.shape[0]), y_pred, color="red",
         linewidth=1.5, linestyle="-.")
plt.legend(['真实值', '预测值'])
plt.show()
