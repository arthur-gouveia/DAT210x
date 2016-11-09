import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper
from sklearn.decomposition import PCA
import sys

# Look pretty...
matplotlib.style.use('ggplot')

# Numerical and Categorical columns
num_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc']
cat_cols = ['classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad',
            'appet', 'pe', 'ane']

# You can set scaleFeatures to True by passing -s as a parameter to the script
if (len(sys.argv) == 2) and (sys.argv[1] == '-s'):
    scaleFeatures = True
else:
    scaleFeatures = False

# TODO: Load up the dataset and remove any and all
# Rows that have a nan. You should be a pro at this
# by now ;-)
#
df = pd.read_csv('Datasets/kidney_disease.csv', index_col=0)
df.dropna(inplace=True)
df = df.reset_index(drop=True)

# Create some color coded labels; the actual label feature
# will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i == 'ckd' else 'green' for i in df.classification]
df.drop('classification', axis=1, inplace=True)

# Make sure that any Object column is coerced to numeric
#
for col in num_cols:
    if df[col].dtype == 'O':
        df[col] = pd.to_numeric(df[col], errors='coerce')


# Change the categorical columns to dummy
df = pd.get_dummies(df)
# Comment the line above and uncomment the line below to see the effect of
# not using dummy variables to discriminate the results
# To see clearly the efect use the -s parameter to scale the variables
# df.drop(cat_cols[1:], axis=1, inplace=True)


# TODO: PCA Operates based on variance. The variable with the greatest
# variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results. Also print out the results of running .describe
# on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then
# you probably didn't complete the previous step properly.
#
print('Old variance:\n', df.var())
print('Old describe:\n', df.describe())


# TODO: This method assumes your dataframe is called df. If it isn't,
# make the appropriate changes. Don't alter the code in scaleFeatures()
# just yet though!
#
if scaleFeatures:
    df = helper.scaleFeatures(df)


# TODO: Run PCA on your dataset and reduce it to 2 components
# Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
pca = PCA(2)
T = pca.fit_transform(df)


# Plot the transformed data as a scatter plot. Recall that transforming
# the data will result in a NumPy NDArray. You can either use MatPlotLib
# to graph it directly, or you can convert it to DataFrame and have pandas
# do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in
# Module4/assignment1.py, this time we'll convert to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we
# are in P.C. space, so we'll just define the coordinates accordingly:
plt.figure()
ax = helper.drawVectors(T, pca.components_, df.columns.values,
                        plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o',
               c=labels, alpha=0.75, ax=ax)
plt.title('Scaled' if scaleFeatures else 'Unscaled')
plt.show()
