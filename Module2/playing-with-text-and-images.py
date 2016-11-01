# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:18:45 2016

@author: Arthur Gouveia

Intro to text, image and sound processing
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy import misc
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavefile
import pandas as pd


def topbar(msg):
    print('\n\n')
    print('='*40)
    print('{:=^40}'.format(' '+msg+' '))
    print('='*40)


def plotimage(subplot, image, title, colmap='jet'):
    plt.subplot(subplot)
    plt.imshow(image, cmap=plt.get_cmap(colmap))
    plt.axis('off')
    plt.title(title)


def plotsound(subplot, data, title):
    plt.subplot(subplot)
    plt.plot(data)
    plt.ylim([-15000, 15000])
    plt.yticks(range(0, 15000, 5000), [])
    plt.xticks(range(0, 140001, 20000), [])
    plt.title(title)


###############################################################################
# Playing with text
###############################################################################

topbar('PLAYING WITH TEXT')

corpus = [
          "Authman ran faster than Harry because he is an athlete.",
          "Authman and Harry ran faster and faster."]

bow = CountVectorizer(ngram_range=(2, 3), stop_words='english')  # Bag Of Words
X = bow.fit_transform(corpus)  # Sparse Matrix
print(bow.get_feature_names())
print('\n', X)
print('\n', X.toarray())


###############################################################################
# Playing with images
###############################################################################

topbar('PLAYING WITH IMAGES')

# Opens 'face' image and normalizes RGB values from 0 to 1
img = misc.face() / 255
print('Original image shape: {}'.format(img.shape))
# Plots the original image
plotimage(221, img, 'Original image {}x{}'.format(img.shape[0], img.shape[1]))
img = img[::2, ::2]
print('Reduced image shape: {}'.format(img.shape))


# Convert the image to grayscale
r, g, b = (img[..., i] for i in range(3))
img = (0.299*r + 0.587*g + 0.114*b)
print('Grayscale image shape: {}'.format(img.shape))

# We need to set the colormap since this is just a lumnosity image
# and plt.imshow uses jet as the default colormap
# http://matplotlib.org/examples/color/colormaps_reference.html
plotimage(223, img,
          'Reduced grayscale image {}x{}'.format(img.shape[0], img.shape[1]),
          colmap='gray')

# Turn the image (2D) into a vector (1D)
img = img.reshape(-1)
# Now img is ready for machine learning or even some data exploration

###############################################################################
# Playing with sound
###############################################################################

topbar('PLAYING WITH SOUND')

# Reading a WAV file
sample_rate, audio_data = wavefile.read('Datasets/Alarm01.wav')
print('\n\nAudio file sample rate: {:0.2f} kbps\n'.format(sample_rate / 1000))

plotsound(222, audio_data[:, 0], 'Audio channel 0')
plotsound(224, audio_data[:, 1], 'Audio channel 1')
