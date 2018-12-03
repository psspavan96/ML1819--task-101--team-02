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
import warnings
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings("ignore")

df = pd.read_csv("../dataset.preprocessed/googleplay/cleaned.csv")
print("Number of data points:",df.shape[0])

from sklearn.model_selection import train_test_split
X = df[['Reviews', 'Size', 'Genres']]
y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

knen = KNeighborsClassifier(n_neighbors=4,algorithm='auto')
knen.fit(X_train_scaled,y_train)
y_pred = knen.predict(X_test_scaled)
print("Accuracy on test set: %0.2f%%"%(accuracy_score(y_test, y_pred)*100))

mse = mean_squared_error(y_test, y_pred)
print("Mse: ", mse)
rms = sqrt(mse)
print("Rmse: ", rms)

