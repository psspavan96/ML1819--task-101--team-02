#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from math import sqrt


df = pd.read_csv("../dataset.preprocessed/googleplay/cleaned.csv")
# print("Number of data points:",df.shape[0])

X = df[['Reviews']]
Y = df['Rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


lm = LinearRegression()

kf = KFold(n_splits=10,shuffle=False)
for train_idx, test_idx in kf.split(X_train, Y_train):
  _x_train = X_train.values[train_idx]
  _y_train = Y_train.values[train_idx]
  lm.fit(X_train, Y_train)

lm.score(X_test, Y_test)

y_pred = lm.predict(X_test) 
plt.plot(Y_test, y_pred, '.')
plt.xlabel('Measured')
plt.ylabel('Predicted')
# plot a line, a perfit predict would all fall on this line
x = np.linspace(0, 10)
y = x
plt.plot(x, y)
plt.show()


mse = mean_squared_error(Y_test, y_pred)
# print("Mean squared error: %.2f"% mse)
# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(Y_test, y_pred))

rms = sqrt(mse)
# print("Rmse: ", rms)
print("rmse,", rms)

mae = mean_absolute_error(Y_test,y_pred)
# print("Mae: ", mae)
