"""
模型解释博客：https://blog.csdn.net/qq_34160248/article/details/127171265
参数说明博客：https://blog.csdn.net/qq_39777550/article/details/109277937
"""
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_iris

# 加载或构造数据
iris = load_iris()
x = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=37)

print('Starting training...')
# 模型训练
gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=20)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)
print('Starting predicting...')
# 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型验证
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
# 特征重要性
print('Feature importances:', list(gbm.feature_importances_))


# 自定义eval_metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Root Mean Squared Logarithmic Error (RMSLE)
def rmsle(y_true, y_pred):
    return 'RMSLE', np.sqrt(np.mean(np.power(np.log1p(y_pred) - np.log1p(y_true), 2))), False


print('Starting training with custom eval function...')
# 模型训练
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=rmsle,
        early_stopping_rounds=5)


# 自定义eval_metric
# f(y_true: array, y_pred: array) -> name: string, eval_result: float, is_higher_better: bool
# Relative Absolute Error (RAE)
def rae(y_true, y_pred):
    return 'RAE', np.sum(np.abs(y_pred - y_true)) / np.sum(np.abs(np.mean(y_true) - y_true)), False


print('Starting training with multiple custom eval functions...')
# 模型训练
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=[rmsle, rae],
        early_stopping_rounds=5)

print('Starting predicting...')
# 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('The rmsle of prediction is:', rmsle(y_test, y_pred)[1])
print('The rae of prediction is:', rae(y_test, y_pred)[1])

# 使用网格搜索优化参数
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid, cv=3)
gbm.fit(X_train, y_train)

print('Best parameters found by grid search are:', gbm.best_params_)
