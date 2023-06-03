from sklearn.decomposition import FastICA, PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# 数据准备
"""
测试数据为三种不同的信号混淆后的数据集（数据集还是N*3大小）
"""
np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)
# 生成3种源信号
waft1 = np.sin(2 * time)  # 正弦信号
waft2 = np.sign(np.sin(3 * time))  # 方波信号
waft3 = signal.sawtooth(2 * np.pi * time)  # 锯齿信号
plt.plot(waft1)
plt.plot(waft2)
plt.plot(waft3)
plt.title("three waft")
plt.show()
# 生成混淆信号
waft = np.c_[waft1, waft2, waft3]
waft += 0.2 * np.random.normal(size=waft.shape)  # 增加噪声
waft /= waft.std(axis=0)  # 数据标准化
arr = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # 混淆矩阵
mix_waft = np.dot(waft, arr.T)  # 生成的混淆信号
plt.plot(mix_waft)
plt.title("mix_waft")
plt.show()
# print(mix_waft)
"""
- 这里说明一下FastICA()对象的常用初始化参数
n_components:   接收int,表示输出的组件数量，若为None，则默认使用所有组件
max_iter:   接收int,表示模型训练的最大迭代次数，默认为200
random_state:   随机数种子
"""
ica = FastICA(n_components=3)  # 初始化FastICA对象
"""
- 这里说明一下FastICA()对象的常用方法
fit(x):   代入数据集x进行训练
fit_transform(x):   训练数据x，并返回降维后的数据集（各个组件数据集）
inverse_transform(x):   将结果x转换为原来的数据集
transform(x):  利用训练好的模型对x进行降维并返回降维结果
"""
ica.fit(mix_waft)
ica_mixing = ica.mixing_
print('ICA使用的混淆矩阵：\n', ica_mixing)
# 使用ICA还原信号
waft_ica = ica.transform(mix_waft)
# 使用PCA还原信号
waft_pca = PCA(n_components=3).fit_transform(mix_waft)
# 绘制结果
plt.figure(figsize=[12, 6])  # 设置画布大小
# 设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
data = [mix_waft, waft, waft_ica, waft_pca]
names = ['混淆信号',
         '实际源信号',
         'ICA复原信号',
         'PCA复原信号']
colors = ['red', 'steelblue', 'orange']
for i, (data, name) in enumerate(zip(data, names), 1):  # 注意一下这里的1
    plt.subplot(4, 1, i)
    plt.title(name)
    for sig, color in zip(data.T, colors):
        plt.plot(sig, color=color)
plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
plt.show()
