from sklearn.preprocessing import OneHotEncoder
import numpy as np

data = np.array([['男', '北京', '已婚'],
                 ['男', '上海', '未婚'],
                 ['女', '广州', '已婚']])
test_data = np.array([['男', '北京', '未婚']])
"""
-------------下面介绍一下几个OneHotEncoder()对象的常用方法-------------
fit():  代入数据训练模型OneHotEncoder()
fit_transform():  先使用fit的方法，在使用transform的方法，即把数据放入后直接返回标准化后的数据

- 在进行.fit()之后，可以调用以下方法：
categories_:  类别
transform:  利用训练好的标准化模型对数据集进行标准化
get_feature_names_out():  返回输出特征的特征名称
inverse_transform:  将独热编码数据变为原来的标签数据
"""

# 创建转换器并生成规则
oh_transfer = OneHotEncoder()
oh_transfer.fit(data)
print(f"独热编码后数据的列标签为：{oh_transfer.get_feature_names_out()}")
print(f"进行独热编码的标签为：{oh_transfer.categories_}")
data_transfer = oh_transfer.transform(data).toarray()
print('独热编码后的训练集为：\n', data_transfer)
print(oh_transfer.inverse_transform(data_transfer))  # 输出原始标签数据
