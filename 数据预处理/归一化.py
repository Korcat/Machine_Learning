from sklearn.preprocessing import Normalizer
from sklearn.datasets import load_iris

data = load_iris().data
norm_transfer = Normalizer()
"""
归一化公式：x' = x / sqrt(sum(x_i^2))
-------------下面介绍一下几个Normalizer()对象的常用方法-------------
fit():  代入数据训练模型
fit_transform:  先使用fit的方法，在使用transform的方法，即把数据放入后直接返回标准化后的数据

- 在进行.fit()之后，可以调用以下方法：
transform:  利用训练好的标准化模型对数据集进行标准化
get_params: 获取模型的参数
"""
norm_transfer.fit(data)  # 代入数据训练模型
print(f"标准化的参数为：{norm_transfer.get_params()}")
data_norm = norm_transfer.transform(data)  # 用训练好的模型对数据进行标准化

# 如果只需要对数据进行标准化，以下步骤更为简洁：
norm_transfer = Normalizer()
data_transfer = norm_transfer.fit_transform(data)  # 调用标准化模型，将数据转化
print(f"标准化后的数据为：{data_transfer}")
