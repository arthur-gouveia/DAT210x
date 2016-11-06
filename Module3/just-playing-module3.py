# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:18:57 2016

Module 4 on DAT210x course scripts

More info on matplotlib histogram:
http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist

@author: Arthur Gouveia
"""

import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.style.use('ggplot')
mpl.cm.cmapname = 'gray'

student_dataset = pd.read_csv("Datasets/students.data", index_col=0)

my_series = student_dataset.G3
my_dataframe = student_dataset[['G3', 'G2', 'G1']]

my_series.plot.hist(alpha=0.5, normed=True)
my_dataframe.plot.hist(alpha=0.5)

student_dataset[['G1', 'G3']].plot.scatter(x='G1', y='G3')

fig = mpl.pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Final Grade')
ax.set_ylabel('First Grade')
ax.set_zlabel('Daily Alcohol')

ax.scatter(student_dataset.G1, student_dataset.G3,
           student_dataset['Dalc'], c='r', marker='o')
mpl.pyplot.show()
