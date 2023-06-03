# coding:utf-8
"""
参数说明博客：https://blog.csdn.net/weixin_41187013/article/details/122615507
模型解释博客：https://blog.csdn.net/weixin_47723732/article/details/122870062
"""

from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_boston
from bayes_opt import BayesianOptimization

# 1.导入数据
boston = load_boston()
x = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)

# 2.参数集定义
param_bounds = {
    'max_depth': (2, 8),
    'n_estimators': (100, 500),
    'learning_rate': (0.1, 0.5),
    "gamma": (0.0, 0.4),
    "reg_alpha": (0.01, 1),
    "reg_lambda": (0.01, 1),
    "min_child_weight": (2, 8),
    "colsample_bytree": (0.6, 0.9),
    "subsample": (0.6, 0.9)
}


# 贝叶斯优化
# 构造黑盒函数
def black_box_function(max_depth, n_estimators, learning_rate, gamma, reg_alpha, reg_lambda,
                       min_child_weight, colsample_bytree, subsample):
    model = XGBRegressor(
        max_depth=int(max_depth),
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        min_child_weight=int(min_child_weight),
        colsample_bytree=colsample_bytree,
        subsample=subsample
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)
    res = mean_squared_error(y_pred, y_train)  # 贝叶斯目标函数为轮廓系数，由于轮廓系数为极小型指标，这里*-1
    return -res


optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=param_bounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)
optimizer.maximize(
    init_points=20,  # 执行随机搜索的步数
    n_iter=20,  # 执行贝叶斯优化的步数
)

# 搜索最优结果
print(optimizer.max)

# 结果展示
best_params = optimizer.max['params']

best_model = XGBRegressor(
    max_depth=int(best_params['max_depth']),
    n_estimators=int(best_params['n_estimators']),
    learning_rate=best_params['learning_rate'],
    gamma=best_params['gamma'],
    reg_alpha=best_params['reg_alpha'],
    reg_lambda=best_params['reg_lambda'],
    min_child_weight=int(best_params['min_child_weight']),
    colsample_bytree=best_params['colsample_bytree'],
    subsample=best_params['subsample']
)
best_model.fit(X_test,y_test)
y_test_pre = best_model.predict(X_test)

# 5.打印测试集RMSE
rmse = sqrt(mean_squared_error(np.array(list(y_test)), np.array(list(y_test_pre))))
print("最优模型在测试集上的rmse为:", rmse)
