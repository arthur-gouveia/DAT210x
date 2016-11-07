import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import parallel_coordinates

# Look pretty...
matplotlib.style.use('ggplot')
norm = input('normalize? [y/n]: ')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
#
df = pd.read_csv('Datasets/wheat.data', index_col=0)



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
#
df.drop(['area', 'perimeter'], axis=1, inplace=True)



#
# TODO: Plot a parallel coordinates chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
#
if norm == 'y':
    df[df.columns[:5]] = df[df.columns[:5]].apply(
                         lambda x: (x - x.min())/(x.max() - x.min()))

plt.figure()

parallel_coordinates(df, 'wheat_type', alpha=0.4)

plt.show()

