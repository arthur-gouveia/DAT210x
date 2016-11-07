import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('seaborn-dark')
#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
#
df = pd.read_csv('Datasets/wheat.data', index_col=0)


#
# TODO: Drop the 'id' feature
#
# already dropped when reading the csv


#
# TODO: Compute the correlation matrix of your dataframe
#
cormatrix = df.corr()


#
# TODO: Graph the correlation matrix using imshow or matshow
#
plt.imshow(cormatrix, cmap=plt.cm.Blues, interpolation='nearest')
plt.colorbar()
tick_marks = [i for i in range(len(df.columns[:7]))]
plt.xticks(tick_marks, df.columns[:7], rotation='vertical')
plt.yticks(tick_marks, df.columns[:7])

plt.matshow(cormatrix, cmap=plt.cm.Reds)
plt.colorbar()
tick_marks = [i for i in range(len(df.columns[:7]))]
plt.xticks(tick_marks, df.columns[:7], rotation='vertical')
plt.yticks(tick_marks, df.columns[:7])

plt.show()
