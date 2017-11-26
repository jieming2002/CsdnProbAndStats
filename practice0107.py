__author__ = 'skye 2017-11-26'
# coding=utf-8
# # 机器学习之概率与统计推断，第一章 随机变量及其分布，（七）案例

import numpy as np
import pandas as pd
from scipy import stats # 分布拟合
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

dpath = './data/'
train = pd.read_json(dpath + 'train.json')
train.head()

print('Train: ', train.shape)

train.info()

train.describe()

# 查看每个变量的分布情况：绘制直方图
sns.countplot(train.interest_level, order=['low', 'medium', 'high'])
plt.xlabel('Interest Level')
plt.ylabel('Number of Occurrences')
plt.show()

# 下面这句没有看到实际的效果
train['interest_level'] = np.where(train.interest_level=='low', 0,
                             np.where(train.interest_level=='medium', 1, 2))

# 绘制散点图
plt.scatter(range(train.shape[0]), train['bathrooms'].values, color='purple')
plt.title('Distribution of Bathrooms')
plt.show()

# 绘制直方图
fig = plt.figure()
# 去除 4 个以上的浴室数据：去除噪声
ulimit = 4
train['bathrooms'].loc[train['bathrooms'] > ulimit] = ulimit

sns.countplot(train.bathrooms)
plt.xlabel('Number of Bathrooms')
plt.ylabel('Number of Occurrences')
plt.show()

# price 特征，散点图
plt.scatter(range(train.shape[0]), train['price'].values, color='purple')
plt.title('Distribution of Price')
plt.show()

# price 去除噪声
ulimit = np.percentile(train.price.values, 99)
# ulimit = 1000000
train['price'].loc[train['price'] > ulimit] = ulimit
# 既可以画直方图，也可以把核密度曲线画出来
sns.distplot(train.price.values, bins=50, kde=True)
plt.xlabel('price', fontsize=12)
plt.show()

# 核密度估计 kde , fit 拟合
sns.distplot(train.price.values, kde=True, fit=stats.norm)
plt.xlabel('price', fontsize=12)
plt.show()

# 极大似然估计 正态分布参数
price_mean = train.price.mean()
price_std = train.price.std()

# 显示估计的正态分布 pdf
# x = train.price
x = np.arange(0, 5*price_std + price_mean, 0.1*price_std)
y = stats.norm.pdf(x, price_mean, price_std)
plt.plot(x, y)
plt.show()

plt.figure()
# 对价格 price 先取 log 再画出直方图。这是经验，直接用不行，就先取 log 就好了
sns.distplot(np.log1p(train['price']))
plt.xlabel('log(price)', fontsize=12)
plt.show()

# 对价格取 log 之后，进行核密度估计 kde , fit 拟合
sns.distplot(np.log1p(train['price']), kde=True, fit=stats.norm)
plt.xlabel('log(price)', fontsize=12)
plt.show()




