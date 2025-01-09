"""
linkage:
    （1）x: 输入的数据可以是待求距离的数据矩阵，也可以是已经求好距离矩阵的下三角的平铺，即一个一维数据
    （2）method: 度量类与类之间类别的方法，complete表示最远距离，single表示最近距离，average表示平均距离
    （3）metric: 如果x的输入为数值矩阵，则该字段接收通过数值矩阵计算距离的方法，有euclidean（欧式距离）、cityblock（曼哈顿距离）、
 cosine（余弦相似度）；如输入的x为一维数据，该字段为None.
dendrogram:
    （1）Z: linkage的输出结果
    （2）p: 后续各个字段可能用到该参数，默认为30.
    （3）truncate_mode: 压缩层次聚类的方法，若为level,则p值为表示层次聚类图显示的深度；若为lastp，则p值为层次聚类显示Z中最后p个类别的层次图
    （4）label: 数据标签，应为传入linkage数值矩阵的列标签（如果传入的为一维数据，则应为平铺之前矩阵的列标签）
    （5）color_threshold: 切割类别的阈值，默认为max(Z[:,2]*0.7；以该值取切割层次聚类图，切到直线的下方所有叶子节点均为一类
    （6）leaf_rotation: 标签逆时针旋转角度
    （7）leaf_font_size: 标签大小

fcluster:
    （1）Z: linkage的返回值
    （2）t: 切割的阈值（等同于层次聚类图中的纵坐标），以t值切割层次聚类图，其与图中类比直线的交点记为返回的类标签
    （3）criterion: 通常为distance
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import warnings

warnings.simplefilter('ignore')

# 导入数据
data_1 = pd.read_excel('聚类数据.xlsx', sheet_name='正常人')
data_1 = data_1.iloc[:, 1:]
data_2 = pd.read_excel('聚类数据.xlsx', sheet_name='糖尿病')
data_2 = data_2.iloc[:, 1:]
data_3 = pd.read_excel('聚类数据.xlsx', sheet_name='高血压')
data_3 = data_3.iloc[:, 1:]

# 标准化
std_transfer1 = StandardScaler()
std_transfer2 = StandardScaler()
std_transfer3 = StandardScaler()

std_transfer1.fit(data_1)
std_transfer2.fit(data_2)
std_transfer3.fit(data_3)

data1_nor = std_transfer1.transform(data_1)
data2_nor = std_transfer2.transform(data_2)
data3_nor = std_transfer3.transform(data_3)

dataset = [data1_nor, data2_nor, data3_nor]
# 设置标签
labels = ['正常人', '糖尿病', '高血压']
Z_list = []
for label, data in zip(labels, dataset):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(20, 8), dpi=300)
    Z = linkage(data, method='ward', metric='euclidean')
    # colors = ['r', 'g', 'b', 'y']  # 自定义颜色列表
    # D = dendrogram(Z, p=10, truncate_mode='lastp')
    D = dendrogram(Z, p=5, truncate_mode='level')
    # D = dendrogram(Z, p=5)
    Z_list.append(Z)
    plt.axhline(y=max(Z[:, 2]) * 0.7, color='pink', linestyle='--')  # 绘制分类切割线
    plt.title(f"{label}层次聚类图")
    # plt.savefig(f"{label}层次聚类图", dpi=300)
    plt.show()

f1 = fcluster(Z_list[0], t=100, criterion='distance')
f2 = fcluster(Z_list[1], t=19, criterion='distance')

print(f1)
