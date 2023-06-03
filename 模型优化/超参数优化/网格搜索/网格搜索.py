from sklearn.model_selection import StratifiedKFold  # 交叉验证
from sklearn.model_selection import GridSearchCV  # 网格搜索
from sklearn.model_selection import train_test_split  # 将数据集分开成训练集和测试集
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris

iris = load_iris()

x = iris.data
y = iris.target

seed = 7  # 重现随机生成的训练
test_size = 0.33  # 33%测试，67%训练
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
model = AdaBoostClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]  # 学习率
n_estimators = [50, 100, 150, 200]

param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators)
# 转化为字典格式，网络搜索要求

kflod = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# 将训练/测试数据集划分10个互斥子集，

grid_search = GridSearchCV(model, param_grid, scoring='neg_log_loss', n_jobs=-1, cv=kflod)
# scoring指定损失函数类型，n_jobs指定全部cpu跑，cv指定交叉验证
grid_result = grid_search.fit(X_train, Y_train)  # 运行网格搜索
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
# grid_scores_：给出不同参数情况下的评价结果。best_params_：描述了已取得最佳结果的参数的组合
# best_score_：成员提供优化过程期间观察到的最好的评分
# 具有键作为列标题和值作为列的dict，可以导入到DataFrame中。
# 注意，“params”键用于存储所有参数候选项的参数设置列表。
means = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']
for mean, param in zip(means, params):
    print("%f  with:   %r" % (mean, param))
