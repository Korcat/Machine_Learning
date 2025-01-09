import numpy as np
"""
模糊聚类的步骤：
1.数据标准化
2.根据某个公式计算每个样本间的相似度，建立模糊相似矩阵
3.计算传递闭包矩阵
4.按lambda从大到小进行聚类（可以画一个聚类动态图）
"""

np.set_printoptions(precision=2)  # 设置矩阵输出精度,保留两位小数


def jmax(a):
    """
    数据标准化
    最大值规格化法
    """
    a = np.array(a)
    c = np.zeros_like(a, dtype=float)
    for j in range(c.shape[1]):  # 遍历c的列
        for i in range(c.shape[0]):  # 遍历c的列
            c[i, j] = a[i, j] / np.max(a[:, j])
    return c


def alike(a):
    """
    模糊相似矩阵
    最大最小法
    """
    a = jmax(a)  # 用标准化后的数据
    c = np.zeros((a.shape[0], a.shape[0]), dtype=float)
    mmax = []
    mmin = []
    for i in range(c.shape[0]):  # 遍历c的行
        for j in range(c.shape[0]):  # 遍历c的行
            mmax.extend([np.fmax(a[i, :], a[j, :])])  # 取i和和j行的最大值,即求i行和j行的并
            mmin.extend([np.fmin(a[i, :], a[j, :])])  # 取i和和j行的最大值,即求i行和j行的交
    for i in range(len(mmax)):
        mmax[i] = np.sum(mmax[i])  # 求并的和
        mmin[i] = np.sum(mmin[i])  # 求交的和
    mmax = np.array(mmax).reshape(c.shape[0], c.shape[1])  # 变换为与c同型的矩阵
    mmin = np.array(mmin).reshape(c.shape[0], c.shape[1])  # 变换为与c同型的矩阵
    for i in range(c.shape[0]):  # 遍历c的行
        for j in range(c.shape[1]):  # 遍历c的列
            c[i, j] = mmin[i, j] / mmax[i, j]  # 赋值相似度
    return c


def hecheng(a, b):
    """
    求模糊是矩阵a和模糊矩阵b的合成
    """
    a, b = np.array(a), np.array(b)
    c = np.zeros_like(a.dot(b))
    for i in range(a.shape[0]):  # 遍历a的行元素
        for j in range(b.shape[1]):  # 遍历b的列元素
            empty = []
            for k in range(a.shape[1]):
                empty.append(min(a[i, k], b[k, j]))  # 行列元素比小
            c[i, j] = max(empty)  # 比小结果取大
    return c


def bibao(a):
    """
    求模糊矩阵a的闭包
    """
    a = alike(a)  # 用模糊相似矩阵
    c = a
    while True:
        m = c
        c = hecheng(hecheng(a, c), hecheng(a, c))
        if (c == m).all():  # 闭包条件
            return np.around(c, decimals=2)  # 返回传递闭包,四舍五入,保留两位小数
            break
        else:
            continue


def julei(a, g):
    a = bibao(a)  # 用传递闭包
    c = np.zeros_like(a, dtype=int)
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            if a[i, j] >= g:
                c[i, j] = 1
            else:
                c[i, j] = 0
    return c


def result(a, g):
    """
    模糊聚类分析结果展示
    """
    a = julei(a, g)
    c = []  # 同类聚合
    for i in range(len(a)):
        x = []
        for j in range(i, len(a)):
            if a[i][j] == 1:
                x.append(j)
            else:
                continue
        c.append(x)
    d = []  # 删除重复1
    for i in range(len(c)):
        for j in range(i + 1, len(c)):
            for k in range(len(c[j])):
                if c[j][k] in c[i]:
                    d.append(c[j])
                    break
                else:
                    continue
    dd = []  # 删除重复2
    for i in range(len(d)):
        for j in range(i + 1, len(d)):
            if d[i] == d[j]:
                dd.append(d[j])
    for i in range(len(dd)):  # 删除重复3
        try:
            d.pop(d.index(dd[i]))
        except ValueError:
            continue
    for i in range(len(d)):  # 删除重复4
        try:
            c.pop(c.index(d[i]))
        except ValueError:
            continue
    for i in range(len(dd)):  # 删除重复5
        try:
            c.pop(c.index(dd[i]))
        except ValueError:
            continue
    for i in range(len(c)):
        for j in range(len(c[i])):
            c[i][j] += 1
    return c


def main():
    """
    测试代码
    """
    x = [[36.3, 29.9, 20.1, 68.9, 70.3, 72],
         [95, 66, 51.6, 54.8, 61, 63.7],
         [10, 8, 8, 74.2, 76.2, 77.1],
         [84.5, 78, 64.8, 54.9, 56.5, 58.9],
         [80, 68, 57.4, 59.1, 62.9, 64.5],
         [60, 36, 26.4, 61.7, 65.8, 68.2],
         [54, 36, 30, 64.8, 68.9, 70.7],
         [10, 5.6, 4.2, 76.6, 79, 80],
         [4.6, 3.2, 2.6, 78.8, 81.1, 82.3],
         [50.5, 37.1, 25.8, 68.3, 65.5, 66.2],
         [42, 42, 42, 69.9, 66.8, 67],
         [8, 5, 4.5, 71.3, 75.9, 78.5],
         [98, 77, 59, 54.6, 60.9, 63.9],
         [16, 11, 9.8, 70.3, 72.6, 74],
         [78.5, 47.6, 34.2, 62.7, 65.1, 67.2],
         [91, 78, 74.4, 59, 60.1, 61.6],
         [95, 85, 77.8, 59.1, 63, 65.2],
         [41, 30, 24, 65.6, 69.6, 71.4],
         [6.7, 2.9, 2.3, 74.3, 78.1, 79.9],
         [25.6, 16.1, 11.2, 71.2, 73.6, 75],
         [25.7, 11.4, 7.2, 67, 68.3, 70.2],
         [38, 23, 14.6, 64.8, 69.1, 70.8],
         [66.7, 40, 28.9, 62.2, 68.8, 71],
         [45, 50, 56, 61.9, 48.5, 50.7],
         [41.5, 31.6, 29.1, 70.9, 74, 74.5],
         [9.4, 6.9, 6.5, 75.2, 77, 77.8],
         [24.7, 16.8, 14.1, 71.7, 73.8, 75],
         [48.1, 26.9, 18.6, 66.6, 70.4, 72.1],
         [26.9, 20.7, 17.7, 71.2, 73.3, 74.4],
         [10.9, 4.1, 3.2, 71.4, 75, 76.5],
         [7.4, 4.4, 3.6, 76.7, 78.9, 80.6],
         [7, 4.4, 3.7, 75.2, 77.9, 79.1],
         [8.2, 4.6, 3.5, 76.9, 79.5, 81.1],
         [7.2, 4.6, 4.2, 76.9, 78, 79.7],
         [19.3, 8.1, 6, 70.9, 73.7, 75.1],
         [22.7, 20.2, 13.7, 68.9, 65.3, 65.6],
         [7.6, 4.5, 3.6, 76.8, 79, 80.8],
         [67, 37.5, 23.7, 66, 70.4, 71.5],
         [21.5, 19.2, 19.8, 70.1, 67.9, 68],
         [8, 5.6, 4.9, 75.9, 77.7, 79.1],
         [8, 4.9, 4.7, 77, 79.2, 81],
         [8.3, 5.9, 5.2, 75.4, 78.6, 79.9]]
    lambdas = np.sort(np.unique(bibao(x)).reshape(-1))[::-1]
    print(lambdas)
    lam_str = ''
    for (i, lam) in zip(range(len(lambdas)), lambdas):
        if i != len(lambdas) - 1:
            lam_str += str(lam) + ' > '
        else:
            lam_str += str(lam)
    print('截集水平：' + lam_str)
    print("原始数据\n", np.array(x))
    print("\n数据标准化(最大值规格化法)\n", jmax(x))
    print("\n模糊相似矩阵(最大最小法)\n", alike(x))
    print("\n传递闭包\n", bibao(x))
    for i in range(0, len(lambdas)):
        g = lambdas[i]
        print("\n模糊聚类分析矩阵(lambda=%0.2f)\n" % g, julei(x, g))
        print("\n模糊聚类结果(lambda=%0.2f)\n" % g, result(x, g))


if __name__ == "__main__":
    main()
