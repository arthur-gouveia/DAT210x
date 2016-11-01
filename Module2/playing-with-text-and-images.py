# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:18:45 2016

@author: Arthur Gouveia

Intro to text, image and sound processing
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy import misc
import matplotlib.pyplot as plt

corpus = [
          "Authman ran faster than Harry because he is an athlete.",
          "Authman and Harry ran faster and faster."]

bow = CountVectorizer(ngram_range=(2, 3), stop_words='english')  # Bag Of Words
X = bow.fit_transform(corpus)  # Sparse Matrix
print(bow.get_feature_names())
print('\n', X)
print('\n', X.toarray())

# Opens 'face' image and normalizes RGB values from 0 to 1
img = misc.face() / 255
print('Original image size: {}'.format(img.shape))
img = img[::2, ::2]
print('Reduced image size: {}'.format(img.shape))

# Convert the image to grayscale
r, g, b = (img[..., i] for i in range(3))
img = (0.299*r + 0.587*g + 0.114*b)
print('Grayscale image size: {}'.format(img.shape))

# We need to set the colormap since this is just a lumnosity image
# and plt.imshow uses jet as the default colormap
# http://matplotlib.org/examples/color/colormaps_reference.html
plt.imshow(img, cmap=plt.set_cmap('gray'))
img = img.reshape(-1)
# Now img is ready for machine learning
