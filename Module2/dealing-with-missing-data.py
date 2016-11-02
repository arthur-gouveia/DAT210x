# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:44:02 2016

JUst playing with missing values

@author: Arthur Gouveia
"""

import numpy as np
import pandas as pd


def gendf(nrows, ncolsunif, ncolsnorm):
    colnames = ['col_{:0>2}'.format(i) for i in range(ncolsunif + ncolsnorm)]
    return pd.DataFrame(np.hstack((np.random.rand(nrows, ncolsunif),
                                   np.random.randn(nrows, ncolsnorm))),
                        columns=colnames)


def setnandf(dframe, col_nans, row_nans):
    # Gets the number of rows and columns
    nrows, ncols = dframe.shape

    nan_cols = np.random.choice(ncols, int(ncols*col_nans), replace=False)
    nan_rows = np.random.choice(nrows, int(nrows*row_nans), replace=False)

    dframe.ix[nan_rows, nan_cols] = np.nan

    return None


def nancolumns(dframe):
    k = np.sum(np.isnan(dframe))
    return [k.index[i] for i in range(len(k)) if k[i] > 0]


if __name__ == '__main__':
    # Generate a random dataframe
    df = gendf(1000, 3, 3)
    setnandf(df, 0.8, 0.3)

    df2 = df.copy() #df2 will be the backup of our dataframe

    print(df.describe())
    print('\n\nMissing values per column:')
    print(np.sum(np.isnan(df)))

    # Columns that have missing values
    nan_columns = nancolumns(df)

    # Replace the first column missing data by the mean
    df[nan_columns[0]] = df[nan_columns[0]].fillna(df[nan_columns[0]].mean())
    # Replace the 2nd column missing values by the previous one
    # Obviously can't replace missing values at the first line
    df[nan_columns[1]] = df[nan_columns[1]].fillna(method='ffill')  # or bfill
    # Interpolate the missing values
    # Can't replace missing values at the start or at the end
    df[nan_columns[2]] = df[nan_columns[2]].interpolate(method='polynomial',
                                                        order=2)
    # remove any column with at least 950 NON missing values
    df = df.dropna(axis=1, thresh=950)
    df = df.dropna(axis=0)  # remove any row with NaN if there's still one

    # This will just remove some coluns
    # df = df.drop(labels=['Features', 'To', 'Delete'], axis=1)

    # Remove duplicates. It checks the subset to see if there are duplicate
    # values. Here I use all columns as subset, but in real world examples we
    # could use an ID column or contract_number
    df = df.drop_duplicates(subset=df.columns)
    # Resets the index after droping lines. drop=True tells pandas no to keep a
    # backup of the index. inplace=True tells pandas to change df rather than
    # returnin a copy of it. This parameter could be used in all functions
    # above. The advantage of inplace=False is to chain methods:
    # df = df.dropna(axis=0, thresh=2).drop(labels=['ColA',
    # axis=1]).drop_duplicates(subset=['ColB', 'ColC']).reset_index()
    df.reset_index(drop=True, inplace=True)
    
    print(df.describe())
