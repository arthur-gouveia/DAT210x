import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from pandas.tools.plotting import andrews_curves

# Look pretty...
matplotlib.style.use('ggplot')
drp = input('Drop features? [y/n]: ')

#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
#
df = pd.read_csv('Datasets/wheat.data', index_col=0)



#
# TODO: Drop the 'id', 'area', and 'perimeter' feature
#
if drp == 'y':
    df.drop(['area', 'perimeter'], axis=1, inplace=True)



#
# TODO: Plot an Andrews Curve chart grouped by
# the 'wheat_type' feature. Be sure to set the optional
# display parameter alpha to 0.4
#

plt.figure()

andrews_curves(df, 'wheat_type', alpha=0.4)

plt.show()

