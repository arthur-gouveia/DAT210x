# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:16:53 2016

@author: Arthur Gouveia

Just playing with pandas
"""

import pandas as pd

tutorial = pd.read_csv('./Datasets/tutorial.csv')
print("{:=^30}".format(' HEAD '))
print(tutorial.head(5))
print("{:=^30}".format(' DESCRIPTION '))
print(tutorial.describe())
print("{:=^30}".format(' INDEX '))
print(tutorial.index)
print("{:=^30}".format(' DTYPES '))
print(tutorial.dtypes)
