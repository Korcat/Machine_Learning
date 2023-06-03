import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV

# 导入训练数据
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

# 分类器使用 xgboost
clf1 = xgb.XGBClassifier()

# 设定搜索的xgboost参数搜索范围，值搜索XGBoost的主要6个参数
param_dist = {
    'n_estimators': range(80, 200, 4),
    'max_depth': range(2, 15, 1),
    'learning_rate': np.linspace(0.01, 2, 20),
    'subsample': np.linspace(0.7, 0.9, 20),
    'colsample_bytree': np.linspace(0.5, 0.98, 10),
    'min_child_weight': range(1, 9, 1)
}

# RandomizedSearchCV参数说明，clf1设置训练的学习器
# param_dist字典类型，放入参数搜索范围
# scoring = 'neg_log_loss'，精度评价方式设定为“neg_log_loss“
# n_iter=300，训练300次，数值越大，获得的参数精度越大，但是搜索时间越长
# n_jobs = -1，使用所有的CPU进行训练，默认为1，使用1个CPU

random_grid = RandomizedSearchCV(clf1, param_dist, cv=5, scoring='neg_log_loss', n_iter=300, n_jobs=-1)

# 在训练集上训练
random_grid.fit(x, y)

# 返回最优的训练器
best_estimator = random_grid.best_estimator_
print(f'best_estimator:{best_estimator}')
print(f'best_param:{random_grid.best_params_}')

# 输出最优训练器的精度
print(random_grid.best_score_)
