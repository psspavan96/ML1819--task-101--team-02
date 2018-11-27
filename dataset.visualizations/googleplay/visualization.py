#!/usr/bin/env python
# coding: utf-8
import seaborn as sns; sns.set(color_codes=True)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../../dataset.preprocessed/cleaned.csv", index_col=0)

g = sns.lmplot(x="Reviews", y="Rating", data=df)
g.savefig('reviews-rating.png')

g = sns.catplot(x="Rated 4.4 or more", y="Reviews", data=df)
g.savefig('rated4.4ormore-reviews.png')

df = df.drop('Rated 4.4 or more', axis=1)
corr = df.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
g= sns.heatmap(corr, cmap=cmap, mask=mask, annot=True, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
g.figure.savefig('heatmap.png')


g = sns.pairplot(df)
g.savefig('pairwise.png')