import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris

"""
x：数据集，二维列表，大小为N*M，即N个数据，M个指标（特征）
y：标签集，一维列表，长度为N，已数字化

"""
data = load_iris()
y = data.target
x = data.data
"""
n_components是PCA类对象的初始化参数之一，输入不同类型的数值可以实现不同的模型训练效果。具体说明如下：
1.为int类型：表示想要保留的主成分个数
2.为float类型：表示输出主成分时需要达到的累计方差占比，其数值应为（0,1）范围的浮点数
3.为str类型：输入降维的模式，如输入“mle”，表示用极大似然估计根据方差分布情况选择合适数量的主成分特征进行降维
"""
n_components = 2  # 手动设置主成分数目
# n_components = 0.95  # 指定降维后各个主成分的方差累计百分比
# n_components = "mle"  # 使用极大似然估计确定最佳主成分个数
pca = PCA(n_components=n_components)  # 加载PCA算法，设置降维后主成分数目为2
reduced_x = pca.fit_transform(x)  # 对样本进行降维,并返回降维后的数据
# inv_x = pca.inverse_transform(reduced_x)  # 将降维后的数据转换成原始数据（inv_x=x）
print(f"降维后的主成分数据集为：{reduced_x}")
# print(inv_x)
# 输出PCA结果
"""
PCA类对象的常用属性：
components_ ：返回具有最大方差的成分。
explained_variance_ratio_：返回 所保留的n个成分各自的方差百分比(不包含除去的成分)
n_components_：返回所保留的成分个数n。
"""
print(f"各主成分的方差解释百分比为：{pca.explained_variance_ratio_}")  # 打印所保留的n个成分各自的方差百分比
# 降维后数据可视化展示
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []
# 因为只有三个类别，这里只有三个循环分支（y=0,1,3）
for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])
# 可视化
plt.scatter(red_x, red_y, c='r', marker='x')
plt.scatter(blue_x, blue_y, c='b', marker='D')
plt.scatter(green_x, green_y, c='g', marker='.')
plt.show()
