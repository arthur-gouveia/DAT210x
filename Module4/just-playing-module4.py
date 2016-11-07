# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 13:20:56 2016

module 3 on DAT210x course scripts

@author: arthu
"""

from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.style.use('seaborn-deep')
df = pd.read_csv('../Module3/Datasets/wheat.data', index_col=0)

df.dropna(inplace=True)

T = PCA(3).fit_transform(df[df.columns[0:-1]])

# Or separating in three steps
# pca = PCA(n_components=3)
# pca.fit(df[df.columns[0:-1]])
# T = pca.transform(df[df.columns[0:-1]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(T[:, 0], T[:, 1], T[:, 2])
plt.show()

pd.tools.plotting.scatter_matrix(df[df.columns[0:-1]])
