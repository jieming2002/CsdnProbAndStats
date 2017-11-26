__author__ = 'skye 2017-11-26'
# coding=utf-8

''' 第一章：随机变量及其分布 课后答疑
关于公式，知道怎么用就可以了。推导过程不必。
K 阶矩，就是 X 的 K 次方，它的期望值。一阶矩就是均值，二阶矩就是方差，三阶矩和斜度有关系。
一阶矩就是随机变量的期望，分布的均值，
二阶矩就是随机变量平方的期望，分布的方差，

阶矩是用来描述随机变量的概率分布的特性.
一阶矩指的是随机变量的平均值,即期望值,
二阶矩指的是随机变量的方差,
三阶矩指的是随机变量的偏度,
四阶矩指的是随机变量的峰度,
因此通过计算矩,则可以得出随机变量的分布形状

关键概念：
均值、方差，描述随机变量分布的最重要的性质。
用样本均值和样本方差，去估计真正分布的均值和方差。

独立同分布，是我们做数据分析的最基本的假设。
只要没有特别说明，我们通常都认为这个数据是独立同分布的样本。
一般，我们在做数据分析时，独立同分布的概念是看不到的，或者说是默认的，不用强调。

总结一下：
描述随机变量，用分布来表示，分布可以用函数 CDF PMF PDF
除了用函数，还可以用典型值来概述分布的性质，概述值有两大类：
一大类是，位置参数：期望/均值、中值/中位数、众数、分位数
第二类是，散布程度：方差、四分位矩
有了数据之后，计算样本均值、样本方差、样本分位数、样本中值。
用直方图、核密度估计分布的类型，

本章研究的是单个随机变量的分布性质。
下一章研究多个随机变量的分布。
'''

''' 作业讲解 '''
# 2. 抛硬币实验：随机采样，以及当样本增多时，频率的稳定性
import numpy as np
import matplotlib.pyplot as plt

def test02Random(N, p):
    # 产生随机数
    s = np.random.random_sample(N) #产生 N 个 [0, 1) 的随机数
    Xi = s <= p     #将随机数变成 Berloulli 分布抽样

    #变成 Binoulli 分布抽样
    Xn = np.zeros(N)
    Pn = np.zeros(N)
    Pn[0] = Xn[0] = Xi[0]
    for i in range(1, N):
        if Xi[i] == True: #正面向上，则计数+1
            Xn[i] = Xn[i-1]+1
        else:#正面向下，则不计数
            Xn[i] = Xn[i-1]
        #正面向上的概率 = 正面向上的次数 / 已经抛硬币次数
        Pn[i] = Xn[i] / float(i+1)

    # 可视化：绘制直方图
    x_grid = range(1, N+1)
    plt.plot(x_grid, Pn, linewidth=2, color='b')
    plt.xlabel('Number of Trial = {0}'.format(N))
    plt.ylabel('Probability = {0}'.format(p))
    plt.show()

def test02Coins():
    # 实验参数
    p = 0.3
    N = 1000
    test02Random(N, p)

    N = 100
    test02Random(N, p)

    N = 10
    test02Random(N, p)

    p = 0.03
    N = 1000
    test02Random(N, p)

    N = 100
    test02Random(N, p)

    N = 10
    test02Random(N, p)

def test03Random(N, B):
    s = np.random.uniform(0, 1, (B, N)) #从均匀分布 [0, 1) 产生 B*N 个样本
    mean_Xn_bar = np.ones(N-5)  #样本均值的均值
    Var_Xn_bar = np.ones(N-5)   #样本均值的方差

    for i in range(5, N):
        si = s[:, :i]
        Xn_bar = np.mean(si, axis=0)  #样本均值
        mean_Xn_bar[i-5] = np.mean(Xn_bar) #得到 B 个（N 个样本均值）的均值
        Var_Xn_bar[i-5] = np.var(Xn_bar)  #得到 B 个（N 个样本均值）的方差

    # 可视化：绘制直方图
    x_grid = range(5, N)
    plt.plot(x_grid, mean_Xn_bar, linewidth=2, color='b')
    plt.xlabel('Number of Trial = {0}'.format(N))
    plt.ylabel('mean_Xn_bar sample = {0}'.format(B))
    plt.show()

    plt.plot(x_grid, Var_Xn_bar, linewidth=2, color='b')
    plt.xlabel('Number of Trial = {0}'.format(N))
    plt.ylabel('Var_Xn_bar sample = {0}'.format(B))
    plt.show()

# 理解抽样分布：大数定理，中心极限定理，均匀分布
def test03Sampling():
    N = 1000 #试验的次数
    B = 100 #样本的个数
    test03Random(N, B)


if __name__ == '__main__':
    # test02Coins()
    test03Sampling()
