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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

df = pd.read_csv("../dataset.preprocessed/googleplay/cleaned.csv")
# print("Number of data points:",df.shape[0])

X = df[['Reviews']]
y = df["Rated 4.4 or more"]
y[y == -1] =0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)

clf = SVC()

kf = KFold(n_splits=10,shuffle=False)
for train_idx, test_idx in kf.split(X_train_scaled,y_train):
  _x_train = X_train_scaled[train_idx]
  _y_train = y_train.values[train_idx]
  clf.fit(_x_train,_y_train)

y_pred = clf.predict(X_test_scaled)
# print("Accuracy on test set: %0.3f"%(accuracy_score(y_test, y_pred)))
print("accuracy,%0.3f"%(accuracy_score(y_test, y_pred)))
# print("F1-Score on test set: %0.2f"%(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)))
