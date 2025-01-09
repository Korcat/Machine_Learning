# 导入所需要的模块
from sklearn import tree  # 树的模块
from sklearn.datasets import load_wine  # 导入红酒数据集
from sklearn.model_selection import train_test_split  # 划分数据集的模块
import pandas as pd
import os
import re

# 探索数据
wine = load_wine()
# 数据有178个样本，13个特征
# wine.data.shape
# 标签
# wine.target
# 如果wine是一张表，应该长这样：
# print(wine)

pd.concat([pd.DataFrame(wine.data), pd.DataFrame(wine.target)], axis=1)
# wine.feature_names
# wine.target_names
# 划分数据为训练集和测试集，test_size标示测试集数据占的百分比
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3)
# Xtrain.shape
# Xtest.shape
"""
参数解释：https://blog.csdn.net/luanpeng825485697/article/details/78965390
"""
# 建立模型
clf = tree.DecisionTreeClassifier(criterion="entropy")  # 实例化模型，添加criterion参数
"""
决策树属性和方法说明
1.属性
feature_importances_：输出每个特征的重要性数值
classes：输出分类模型的分类标签
2.方法
fit(x):   代入数据集x进行训练
apply(x):  返回每个样本叶结点的索引
decision_path(x):   返回决策路径   
predict(x): 输出样本数据x的类别标签
predict_proba(x):  计算x的概率,返回的是一个N*K的二维列表，K为特征个数
score(x,y): 返回测试数据集的平均准确度，x为特征数据集，y为x对应的真实标签
"""
clf = clf.fit(Xtrain, Ytrain)  # 使用实例化好的模型进行拟合操作
score = clf.score(Xtest, Ytest)  # 返回预测的准确度
print(score)
# 在这里，我们发现每一次运行程序时，返回的准确率都不相同，这是因为sklearn每次都在全部的特征中选取若干个特征建立一棵树
# 最后选择准确率最高的决策树返回，如果我们添加上random_state参数，那么sklearn每一次建立决策树使用的特征都相同，返回的预测分数也会一样

# random_state是决定随机数的参数，随机数不变，那么每一次创建的决策树也一样

# 对数据进行可视化
feature_name = wine.feature_names
# feature_name = ['酒精', '苹果酸', '灰', '灰的碱性', '镁', '总酚', '类黄酮', '非黄烷类酚类', '花青素', '颜色强度', '色调', '稀释葡萄酒', '脯氨酸']

# 以下这两行是手动进行环境变量配置，防止在本机环境的变量部署失败
# os.environ['PATH'] = os.pathsep + r'D:\Graphviz\bin'

dot_data = tree.export_graphviz(clf  # 训练好的模型
                                , out_file=None
                                , feature_names=feature_name
                                , class_names=["琴酒", "雪莉", "贝尔摩德"]  # 这里的名字应为对应编码的名字，如琴酒-0，雪莉-1，贝尔摩德-2
                                , filled=True  # 进行颜色填充
                                , rounded=True  # 树节点的形状控制
                                )
# 以下方法生成图像有乱码，不采用
# graph = graphviz.Source(dot_data)
# graph.render("决策树可视化")

# 将生成的dot_data内容导入到txt文件中
f = open('dot_data.txt', 'w')
f.write(dot_data)
f.close()
# 修改字体设置，避免中文乱码！

f_old = open('dot_data.txt', 'r')
f_new = open('dot_data_new.txt', 'w', encoding='utf-8')
for line in f_old:
    if 'fontname' in line:
        font_re = 'fontname=(.*?)]'
        old_font = re.findall(font_re, line)[0]
        line = line.replace(old_font, 'SimHei')
    f_new.write(line)
f_old.close()
f_new.close()
# 以PNG的图片形式存储生成的可视化文件
os.system('dot -Tpng dot_data_new.txt -o 决策树模型.png')
print('决策树模型.png已经保存在代码所在文件夹！')
# 以PDF的形式存储生成的可视化文件
os.system('dot -Tpdf dot_data_new.txt -o 决策树模型.pdf')
print('决策树模型.pdf已经保存在代码所在文件夹！')

# 特征重要性
# clf.feature_importances_  # 查看每一个特征对分类的贡献率
print(f'各个特征的重要性为：{[*zip(feature_name, clf.feature_importances_)]}\n')
print(f'类别标签为：{clf.classes_}\n')

# 在这里，我们发现每一次运行程序时，返回的准确率都不相同，这是因为sklearn每次都在全部的特征中选取若干个特征建立一棵树
# 最后选择准确率最高的决策树返回，如果我们添加上random_state参数，那么sklearn每一次建立决策树使用的特征都相同，返回的
# 预测分数也会一样clf = tree.DecisionTreeClassifier(criterion="entropy",random_state=30)
clf = clf.fit(Xtrain, Ytrain)
score = clf.score(Xtest, Ytest)  # 返回预测的准确度
print(score)
