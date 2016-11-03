import pandas as pd
#import numpy as np

#
# TODO:
# Load up the dataset, setting correct header labels.
#
census = pd.read_csv('Datasets/census.data',
                     index_col=0,
                     names=['education', 'age', 'capital-gain', 'race',
                            'capital-loss', 'hours-per-week', 'sex',
                            'classification'],
                     na_values='Preschool')

#
# TODO:
# Use basic pandas commands to look through the dataset... get a
# feel for it before proceeding! Do the data-types of each column
# reflect the values you see when you look through the data using
# a text editor / spread sheet program? If you see 'object' where
# you expect to see 'int32' / 'float64', that is a good indicator
# that there is probably a string or missing value in a column.
# use `your_data_frame['your_column'].unique()` to see the unique
# values of each column and identify the rogue values. If these
# should be represented as nans, you can convert them using
# na_values when loading the dataframe.
#
print('Original dtypes')
print(census.dtypes)
census['capital-gain'] = pd.to_numeric(census['capital-gain'], errors='coerce')
print('\n\nCorrected dtypes')
print(census.dtypes)

is_str_col = [type(census.loc[0, col]) == str for col in census.columns]
str_cols = census.columns[is_str_col]
num_cols = census.columns[[not(x) for x in is_str_col]]

# Numeric summary of numerical columns
print(census[num_cols].describe())
# Frequency table for nominal columns
for col in str_cols:
    print('\nCount of unique values of "{}":\n{}'.format(col,
                                                         census[col].
                                                         value_counts()))

# Things that called my attentions:
#   1 - 51 people declared 'Preschool' as education level
#   2 - min and max hours per week: 1 and 99
#   3 - max capital-gain: 99999
#   4 - ages == 90

#
# TODO:
# Look through your data and identify any potential categorical
# features. Ensure you properly encode any ordinal and nominal
# types using the methods discussed in the chapter.
#
# Be careful! Some features can be represented as either categorical
# or continuous (numerical). Think to yourself, does it generally
# make more sense to have a numeric type or a series of categories
# for these somewhat ambigious features?
#
# Ordinal categories

params = ('ordered=True', 'categories=set(census.classification)')
census.classification = census.classification.astype("category",
                                                     *params).cat.codes
education_categories = ['1st-4th',
                        '5th-6th',
                        '7th-8th',
                        '9th',
                        '10th',
                        '11th',
                        '12th',
                        'HS-grad',
                        'Some-college',
                        'Bachelors',
                        'Masters',
                        'Doctorate']

census.education = census.education.astype('category', ordered=True,
                                           categories=education_categories
                                           ).cat.codes

# Nominal categories

census = pd.get_dummies(census, columns=['sex', 'race'])

#
# TODO:
# Print out your dataframe
#
# .. your code here ..
print(census)
