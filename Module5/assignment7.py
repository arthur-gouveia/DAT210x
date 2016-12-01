import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn import preprocessing as skpre
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = True


def plotDecisionBoundary(model, X, y):
    print("Plotting...")
    import matplotlib
    matplotlib.style.use('ggplot')  # Look Pretty

    fig = plt.figure()
    fig.add_subplot(111)

    padding = 0.1
    resolution = 0.1

    # (2 for benign, 4 for malignant)
    colors = {2: 'royalblue', 4: 'lightsalmon'}

    # Calculate the boundaris
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * padding
    y_min -= y_range * padding
    x_max += x_range * padding
    y_max += y_range * padding

    # Create a 2D Grid Matrix. The values stored in the matrix
    # are the predictions of the class at at said location
    import numpy as np
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))

    # What class does the classifier say?
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour map
    plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
    plt.axis('tight')

    # Plot your testing points as well...
    for label in np.unique(y):
        indices = np.where(y == label)
        plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

    p = model.get_params()
    plt.title('K = ' + str(p['n_neighbors']))
    plt.show()


#
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
# .. your code here ..
df_columns = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial',
              'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status']
df = pd.read_csv('Datasets/breast-cancer-wisconsin.data', index_col=0,
                 names=df_columns, na_values='?')
df.nuclei.fillna(0, inplace=True)

#
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. You can also drop the sample column, since that doesn't provide
# us with any machine learning power.
#
# .. your code here ..
status = df.status.copy()
df.drop('status', axis=1, inplace=True)


#
# TODO: With the labels safely extracted from the dataset, replace any nan
# values with the mean feature / column value
#
# .. your code here ..
# No missing values!
# You could check with df.isnull.sum()

# Checking data distribution
# df.hist()
# plt.figure()
# df.boxplot()

#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation.
#
# .. your code here ..
# Normalizer()
# MinMaxScaler()
# RobustScaler()
# df = skpre.RobustScaler().fit_transform(df)

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
    print("Computing 2D Principle Components")
    #
    # TODO: Implement PCA here. save your model into the variable 'model'.
    # You should reduce down to two dimensions.
    #
    # .. your code here ..
    model = PCA(2)


else:
    print("Computing 2D Isomap Manifold")
    #
    # TODO: Implement Isomap here. save your model into the variable 'model'
    # Experiment with K values from 5-10.
    # You should reduce down to two dimensions.
    #
    # .. your code here ..
    model = Isomap(n_neighbors=6, n_components=2)

#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform
# in the reading material, but set the random_state=7 for reproduceability, and
# keep the test_size at 0.5 (50%).
#
# .. your code here ..
X_train, X_test, y_train, y_test = train_test_split(df, status, test_size=0.5,
                                                    random_state=7)


#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
# .. your code here ..
model.fit(X_train)
X_train = model.transform(X_train)
X_test = model.transform(X_test)


#
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# .. your code here ..
max_score = 0
best_k = 0
best_weight = ''
score = 0

for k in range(1, 15):
    for weight in ['uniform', 'distance']:
        knmodel = KNeighborsClassifier(n_neighbors=k,
                                       weights=weight).fit(X_train, y_train)
        score = knmodel.score(X_test, y_test)
        if score > max_score:
            best_k = k
            best_weight = weight
            max_score = score

knmodel = KNeighborsClassifier(n_neighbors=best_k,
                               weights=best_weight).fit(X_train, y_train)
#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the
# UDF weights don't give you any class information, the only way to introduce
# this data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.

#
# TODO: Calculate + Print the accuracy of the testing set
#
# .. your code here ..

print('Best k: {}\nBest weight: {}\nBest score: {:.4f}%'.format(
      best_k, best_weight, max_score*100))
plotDecisionBoundary(knmodel, X_test, y_test)
