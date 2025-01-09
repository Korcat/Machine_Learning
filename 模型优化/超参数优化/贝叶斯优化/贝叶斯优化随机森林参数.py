from sklearn.ensemble import RandomForestRegressor
from bayes_opt import BayesianOptimization  # pytorch
import pandas as pd
from sklearn.model_selection import train_test_split

# 贝叶斯优化随机森林参数实例

### 测试准备
# 导入数据（鸢尾花数据集）
data = pd.read_csv('data/iris.csv')
iris_types = ['SETOSA', 'VERSICOLOR', 'VIRGINICA']
col = data.columns[0:4]  # 提取数据集的指标
dv = data.values  # 获取pd数据集的值
X = dv[:, 0:4]  # 自变量
Y = dv[:, 4]  # 因变量
for i in range(0, Y.shape[0]):
    Y[i] = iris_types.index(Y[i])  # 将因变量的值转化为数值类型

x_train, x_test, y_train, y_test = train_test_split(X, Y)  # 划分训练集和测试集
print(y_train)
# x_axis = 'petal_length'
# y_axis = 'petal_width'


# for iris_type in iris_types:
#     plt.scatter(data[x_axis][data['class'] == iris_type],
#                 data[y_axis][data['class'] == iris_type],
#                 label=iris_type
#                 )
# plt.show()


# 测试原参数下的分类效果
model = RandomForestRegressor()
model.fit(x_train, y_train)
origin_score = model.score(x_test, y_test)

# print('默认参数下测试集评分：')
print(origin_score)


### 贝叶斯优化
# 1.构造黑盒函数
def black_box_function(n_estimators, min_samples_split, max_features, max_depth):
    rf = RandomForestRegressor(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),  # float
                               max_depth=int(max_depth),
                               random_state=2
                               )
    rf.fit(x_train, y_train)
    res = rf.score(x_test, y_test)  # 贝叶斯优化要有目标函数
    return res


# 2.确定域空间
pbounds = {'n_estimators': (10, 250),
           'min_samples_split': (2, 25),
           'max_features': (0.1, 0.999),
           'max_depth': (5, 15)}

# 3.实例化对象
optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    random_state=1,
)

# 4.确定迭代次数
optimizer.maximize(
    init_points=5,  # 执行随机搜索的步数
    n_iter=25,  # 执行贝叶斯优化的步数
)

# 5.搜索最优结果
print(optimizer.max)

# 6.优化结果比较
bayes_score = optimizer.max['target']
if bayes_score > origin_score:
    print(f"贝叶斯优化有效！得分优化了{bayes_score - origin_score}")
else:
    print("贝叶斯优化效果不佳")
