from sklearn.datasets import load_iris
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV

# 1.加载数据集
iris = load_iris()
X = iris.data
Y = iris.target
"""
鸢尾花数据集参数解释：
iris["data"]:花的特征数据，是一个N*4的二维列表（N是数据集的个数，4表示每一个花都有四个特征）
iris.target:花的类型，是一个一维列表，数字范围为0-3
"""
# print("鸢尾花数据集的返回值：\n", iris)
# # 返回值是一个继承自字典的Bench
# print("鸢尾花的特征值:\n", iris["data"])
# print("鸢尾花的目标值：\n", iris.target)
# print("鸢尾花特征的名字：\n", iris.feature_names)
# print("鸢尾花目标值的名字：\n", iris.target_names)
# print("鸢尾花的描述：\n", iris.DESCR)


# 数据展示
df = pd.DataFrame(X, columns=['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width'])
df['Species'] = Y


def plot_iris(iris, col1, col2):
    sns.lmplot(x=col1, y=col2, data=iris, hue="Species", fit_reg=False)
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title('data present')
    plt.show()


plot_iris(df, 'Petal_Width', 'Sepal_Length')

# 2.数据基本处理
"""
X：数据集，一行就是一个对象数据，其每一列为该对象数据的不同特征
Y：标签集，都已数字化，是一个一维列表
"""
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=22)  # 测试用
# 实际训练用
# x_train = X
# y_train = Y
# x_test=

# 3、特征工程：标准化
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
# x_test = transfer.transform(x_test)

# 4、模型训练
"""
KNN()对象初始化参数
n_neighbors：KNN中的k值，默认为5（对于k值的选择，前面已经给出解释）；
weights：用于标识每个样本的近邻样本的权重，可选择"uniform",“distance” 或自定义权重。默认"uniform"，所有最近邻样本权重都一样。如果是"distance"，则权重和距离成反比例；如果样本的分布是比较成簇的，即各类样本都在相对分开的簇中时，我们用默认的"uniform"就可以了，如果样本的分布比较乱，规律不好寻找，选择"distance"是一个比较好的选择；
algorithm：接收str,限定半径最近邻法使用的算法，可选‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
"""
knn = KNN(algorithm="kd_tree")  # 模型准备
# 使用网格搜索确定最佳k值
"""
输入：
    estimator：估计器对象
    param_grid：估计器参数(dict){“n_neighbors”:[1,3,5]}
    cv：指定几折交叉验证
    fit：输入训练数据
    score：准确率
输出：
    best_params_：最好的模型参数（搜索参数）
"""
param_dict = {"n_neighbors": [i for i in range(1, 10)]}  # 准备要调的超参数
model = GridSearchCV(knn, param_grid=param_dict, cv=3)
model.fit(x_train, y_train)  # 训练

# 5、评估模型效果
# 比对预测结果和真实值
y_predict = model.predict(x_test)  # 获得预测标签
# print("比对预测结果和真实值：\n", y_predict == y_test)
# 计算准确率
# score = model.score(x_test, y_test)
# print("直接计算准确率：\n", score)

# 打印最优模型参数
print(model.best_params_)
