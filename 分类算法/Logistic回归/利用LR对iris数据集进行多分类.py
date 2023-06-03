from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR

# 导入数据
iris = load_iris()
x = iris.data
y = iris.target
# 数据预处理（标准化）
std_transfer = StandardScaler()  # 生成标准化模型对象
x_std = std_transfer.fit_transform(x)
# 划分数据集，分为测试集合验证集
x_train, x_test, y_train, y_test = train_test_split(x_std, y,
                                                    test_size=0.2, random_state=22)
# LR模型构建
lr1 = LR(multi_class="multinomial", max_iter=100, solver="saga")  # 多分类模型
lr2 = LR(multi_class="ovr", max_iter=100)  # 二分类模型
# 模型训练
lr1.fit(x_train, y_train)
lr2.fit(x_train, y_train)
# 输出分类结果
print(f'多分类的预测准确度为：{lr1.score(x_test, y_test)}')
print(f'多分类前十个预测结果为：{lr1.predict(x_test[:10])}')
print(f'多分类前十个实际结果为：{y_test[:10]}')
print(f'多分类前十个预测概率为：{lr1.predict_proba(x_test[:10])}')
print(f'二分类的预测准确度为：{lr2.score(x_test, y_test)}')
print(f'二分类前十个的预测结果为：{lr2.predict(x_test[:10])}')
print(f'二分类前十个的实际结果为：{y_test[:10]}')
print(f'二分类前十个预测概率为：{lr2.predict_proba(x_test[:10])}')
