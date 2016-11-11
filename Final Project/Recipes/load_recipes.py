# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 02:14:15 2016

@author: Arthur Gouveia
"""

import urllib
from bs4 import BeautifulSoup
import pandas as pd


def get_recipe(url):
    r = urllib.request.urlopen(url).read()

    soup = BeautifulSoup(r, 'html.parser')

    recipe = soup.title.string.split(' Recipe')[0]
    ingredients = [ing.string
                   for ing in soup.find_all('span',
                                            class_='recipe-ingred_txt added')]
    directions = [direc.string for direc in soup.find_all('li', class_='step')]
    return (recipe, ingredients, directions)

url_base = 'http://allrecipes.com/recipe/'

recipes = []
ingredients = []
directions = []
for i in range(10000, 10010):
    recipe_name, ing_list, direc_list = get_recipe(url_base+str(i))
    recipes.append(recipe_name)
    ingredients.append(ing_list)
    directions.append(direc_list)

recipesdf = pd.DataFrame([ingredients, directions])
recipesdf = recipesdf.T
recipesdf.index = recipes
recipesdf.columns = ['Ingredients', 'Directions']
