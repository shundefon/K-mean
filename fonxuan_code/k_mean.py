import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.rcParams['font.family'] = ['Microsoft YaHei'] # 设置字体为中文
plt.rcParams['font.size'] = 12 # 增加字体大小

# 读取数据
df = pd.read_spss('C:/Users/fzx/Desktop/data.sav')

# 提取X变量并进行标准化
X = df[['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']]
scaler = StandardScaler()
X_std = scaler.fit_transform(X)
X_agg = np.sum(X_std, axis=1)
df['X_agg'] = X_agg

# 求和得到新的聚合变量
X_agg = np.sum(X_std, axis=1)
# 计算距离矩阵
dists = linkage(X_std, method='ward', metric='euclidean')

# 进行KMeans聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(X_std)


# 绘制谱系图
plt.figure(figsize=(12, 8))
plt.title('谱系图')
plt.xlabel('欧氏距离')
plt.ylabel('城市')
plt.xticks(rotation=90)
dendrogram(dists, labels=df['city'].values, orientation='left')
#plt.yticks(range(len(df['city'])), df['city'], fontsize=20) # 减小标签字体大小
plt.tight_layout() # 调整图形布局以避免重叠
plt.show()
# 计算相关系数矩阵
corr_matrix = df.corr()

# 绘制相关系数矩阵热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')

# 绘制带误差线的冰柱图
plt.figure(figsize=(10, 8))
colors = {0: 'r', 1: 'g', 2: 'b'}
for i, city in enumerate(df['city']):
    plt.bar(i, df.loc[i, 'X_agg'], yerr=df.loc[i, 'X_agg'], color=colors[labels[i]])
plt.xticks(range(len(df['city'])), df['city'], fontsize=8,rotation=90)
plt.title('Bar Chart with Error Bars')
plt.legend(['Cluster 0', 'Cluster 1', 'Cluster 2'])
plt.show()

