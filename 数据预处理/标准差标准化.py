from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

data = load_iris().data
std_transfer = StandardScaler()  # 生成标准化模型对象
"""
标准化公式：x' = (x-μ) / σ
-------------下面介绍一下几个StandardScaler()对象的常用方法-------------
fit():  代入数据训练模型
fit_transform():  先使用fit的方法，在使用transform的方法，即把数据放入后直接返回标准化后的数据

- 在进行.fit()之后，可以调用以下方法：
.mean_:  每个特征的均值
.var_:  每个特征的方差
transform:  利用训练好的标准化模型对数据集进行标准化
get_params: 获取模型的参数
inverse_transform:  逆标准化，即将标准化后的数据集变为原来未处理的形式
"""
std_transfer.fit(data)  # 代入数据训练模型
print(f"标准化的均值为：{std_transfer.mean_}")
print(f"标准化的方差分为：{std_transfer.var_}")
print(f"标准化的参数为：{std_transfer.get_params()}")
data_std = std_transfer.transform(data)  # 用训练好的模型对数据进行标准化

# 如果只需要对数据进行标准化，以下步骤更为简洁：
std_transfer = StandardScaler()
data_transfer = std_transfer.fit_transform(data)  # 调用标准化模型，将数据转化
print(f"标准化后的数据为：{data_transfer}")
