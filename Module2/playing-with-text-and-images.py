# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:18:45 2016

@author: Arthur Gouveia

Intro to text, image and sound processing
"""

from sklearn.feature_extraction.text import CountVectorizer

corpus = [
          "Authman ran faster than Harry because he is an athlete.",
          "Authman and Harry ran faster and faster."]

bow = CountVectorizer(ngram_range=(2,3))  # Bag Of Words
X = bow.fit_transform(corpus)  # Sparse Matrix
print(bow.get_feature_names())
print('\n', X)
print('\n', X.toarray())
