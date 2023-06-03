# 均方差
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

boston = load_boston()
x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)
ridge_model = Ridge()
ridge_model.fit(x_train, y_train)
y_pred = ridge_model.predict(x_test)

print('均方差为：', mean_squared_error(y_true=y_test, y_pred=y_pred))

# 平均绝对误差
print('平均绝对误差为：', mean_absolute_error(y_true=y_test, y_pred=y_pred))

# 中值绝对误差
print('中值绝对误差为：', median_absolute_error(y_true=y_test, y_pred=y_pred))
print('R2决定系数为：', r2_score(y_true=y_test, y_pred=y_pred))
