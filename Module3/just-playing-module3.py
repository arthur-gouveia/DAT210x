# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 13:18:57 2016

Module 3 on DAT210x course scripts

More info on matplotlib histogram:
http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.hist

@author: Arthur Gouveia
"""


def histplot(data, **kwargs):
    data.plot.hist(**kwargs)


def scatter2D(data, x, y, **kwargs):
    data.plot.scatter(x=x, y=y, **kwargs)


def scatter3D(data, **kwargs):
    fig = mpl.pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Final Grade')
    ax.set_ylabel('First Grade')
    ax.set_zlabel('Daily Alcohol')

    ax.scatter(data[0], data[1],
               data[2], **kwargs)
    mpl.pyplot.show()


import pandas as pd
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

mpl.style.use('ggplot')
mpl.cm.cmapname = 'gray'

student_dataset = pd.read_csv("Datasets/students.data", index_col=0)

histplot(student_dataset.G3, alpha=0.5, normed=True)
histplot(student_dataset[['G3', 'G2', 'G1']], alpha=0.5)

scatter2D(student_dataset[['G1', 'G3']], x='G1', y='G3')

scatter3D([student_dataset.G1, student_dataset.G3,
           student_dataset['Dalc']], c='r', marker='o')
