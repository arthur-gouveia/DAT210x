import pandas as pd
import os
from sklearn import manifold
from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
root = 'Datasets/ALOI/32/'

samples = [misc.imread(root+file).reshape(-1) for file in os.listdir(root)]


#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.
#
# .. your code here .. 


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 


#
# TODO: Convert the list to a dataframe
#
df = pd.DataFrame(samples)


#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
iso_t = manifold.Isomap(n_neighbors=6, n_components=3).fit_transform(df)


#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
plt.scatter(iso_t[:, 0], iso_t[:, 1])


#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Isomap 3D')
ax.set_xlabel('Component 0')
ax.set_ylabel('Component 1')
ax.set_zlabel('Component 2')
ax.scatter(iso_t[:, 0], iso_t[:, 1], iso_t[:, 2],
           c='green', marker='.', alpha=0.75)




plt.show()

