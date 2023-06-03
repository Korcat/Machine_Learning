from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target
# 构建并训练LDA模型
"""
- LDA()对象初始化参数说明
n_components：降维后的维数，只接收[1,k-1]范围内的整数，k为原始数据集的特征数
"""
lda = LDA(n_components=2)
"""
LDA()对象调用方法及属性说明
1.属性：
explained_variance_ratio_:  表示降维后的方差百分比
classes_:   模型输出的样本类标签
2.方法
fit(x):   代入数据集x进行训练
fit_transform(x):   训练数据x，并返回降维后的数据集（各个组件数据集）
transform(x):  利用训练好的模型对x进行降维并返回降维结果
predict(x): 输出样本数据x的类别标签
predict_proba(x):  计算x的概率,返回的是一个N*K的二维列表，K为特征个数
score(x,y): 返回测试数据集的准确度，x为特征数据集，y为x对应的真实标签
"""
lda.fit(x, y)  # 注意LDA需要输入两组数据（特征数据集合标签数据集）

# 构建并训练PCA模型
pca = PCA(n_components=2)
pca.fit(x)

print('LDA模型方差百分比为：', lda.explained_variance_ratio_)
print('LDA模型类标签为：', lda.classes_)
print(f'LDA模型的准确度为：{lda.score(x,y)}')
print(f'LDA模型的预测概率为：{lda.predict_proba(x)}')

# 获取LDA与PCA模型的降维结果
target_names = iris.target_names
x_lda = lda.transform(x)
x_pca = pca.transform(x)
# 绘制图形进行效果对比
plt.figure()
# 设置中文显示
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
colors = ['navy', 'turquoise', 'darkorange']
markers = ['*', '.', 'd']
lw = 2
for color, i, target_name, marker in zip(colors, [0, 1, 2],
                                         target_names, markers):
    plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], color=color,
                alpha=.8, lw=lw, label=target_name,
                marker=marker)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA降维结果')
plt.figure()
for color, i, target_name, marker in zip(colors, [0, 1, 2],
                                         target_names, markers):
    plt.scatter(x_lda[y == i, 0], x_lda[y == i, 1],
                alpha=.8, color=color, label=target_name,
                marker=marker)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA降维结果')
plt.show()
