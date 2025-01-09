import matplotlib.pyplot as plt  # 加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

"""
x：数据集，二维列表，大小为N*M，即N个数据，M个指标（特征）
y：标签集，一维列表，长度为N，已数字化

"""
data = load_iris()
y = data.target
x = data.data
# 数据标准化
std_transfer = StandardScaler()  # 生成标准化模型对象
std_transfer.fit(x)  # 代入数据训练模型
x_std = std_transfer.transform(x)  # 用训练好的模型对数据进行标准化
"""
n_components是PCA类对象的初始化参数之一，输入不同类型的数值可以实现不同的模型训练效果。具体说明如下：
1.为int类型：表示想要保留的主成分个数
2.为float类型：表示输出主成分时需要达到的累计方差占比，其数值应为（0,1）范围的浮点数
3.为str类型：输入降维的模式，如输入“mle”，表示用极大似然估计根据方差分布情况选择合适数量的主成分特征进行降维
"""

pca = PCA()  # 不指定主成分的个数
reduced_x = pca.fit_transform(x_std)  # 对样本进行降维,并返回降维后的数据
explained_variance_ratio_list = pca.explained_variance_ratio_
accumulate_list = [sum(explained_variance_ratio_list[:i]) for i in range(1, len(explained_variance_ratio_list) + 1)]
print(explained_variance_ratio_list)

# 绘制落石图确定主成分个数
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决“-”显示异常
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # 创建与ax1共享x轴的第二个y轴

ax1.plot(range(1, len(explained_variance_ratio_list) + 1), explained_variance_ratio_list, 'bd-.', label='特征方差占比')
ax1.set_xlabel('主成分个数')
ax1.set_xticks(range(1, len(explained_variance_ratio_list) + 1))
ax1.set_ylabel('特征方差占比')

ax2.plot(range(1, len(explained_variance_ratio_list) + 1), accumulate_list, 'rs-', label='特征方差累计占比')
ax2.set_ylabel('特征方差累计占比')

# 将label统一
lines = [ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels()]
handles, labels = [sum(lol, []) for lol in zip(*lines)]

plt.legend(handles, labels, loc='center right')
plt.show()
