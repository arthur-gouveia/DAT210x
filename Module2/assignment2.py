import pandas as pd

# TODO: Load up the 'tutorial.csv' dataset

tutorial = pd.read_csv('Datasets/tutorial.csv')

# TODO: Print the results of the .describe() method

print(tutorial.describe())

# TODO: Figure out which indexing method you need to
# use in order to index your dataframe with: [2:4,'col3']
# And print the results

print(tutorial.ix[2:4, 'col3'])
