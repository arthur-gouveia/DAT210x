# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 15:16:53 2016

@author: Arthur Gouveia

Just playing with pandas
"""

import pandas as pd

tutorial = pd.read_csv('./Datasets/tutorial.csv')
print("{:=^40}".format(' HEAD '))
print(tutorial.head(5))
print("{:=^40}".format(' DESCRIPTION '))
print(tutorial.describe())
print("{:=^40}".format(' INDEX '))
print(tutorial.index)
print("{:=^40}".format(' DTYPES '))
print(tutorial.dtypes)

print('='*40)
print('{:=^40}'.format(' Changing text to categorical int '))
print('='*40)
ordered_satisfaction = ['Very Unhappy', 'Unhappy',
                        'Neutral', 'Happy', 'Very Happy']
df = pd.DataFrame({'satisfaction': ['Mad', 'Happy', 'Unhappy', 'Neutral']})
print('\nSatisfaction as text')
print(df)
df.satisfaction = df.satisfaction.astype("category", ordered=True,
                                         categories=ordered_satisfaction
                                         ).cat.codes

print('\nSatisfaction as int category')
print(df)

df2 = pd.DataFrame({'vertebrates': ['Bird',
                                    'Bird',
                                    'Mammal',
                                    'Fish',
                                    'Amphibian',
                                    'Reptile',
                                    'Mammal',
                                    ]})

print('\nVertebrates as text')
print(df2)
# Method 1)
print('\nVertebrates as int')
print(df2.vertebrates.astype("category").cat.codes)

# Method 2)
print('\nvertebrates as dummy columns')
print(pd.get_dummies(df2, columns=['vertebrates']))
