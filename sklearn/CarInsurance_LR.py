#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm,tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("../dataset.preprocessed/carinsurance/cleaned.csv")
# print("Number of data points:",df.shape[0])

X = df[['NoOfContacts']]
Y = df['LastContactDay']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)


lm = LinearRegression()

kf = KFold(n_splits=10,shuffle=True)
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
x = np.linspace(0, 30)
y = x
plt.plot(x, y)
plt.show()

# print("Mean squared error: %.2f"% mean_squared_error(Y_test, y_pred))
# Explained variance score: 1 is perfect prediction
# print('Variance score: %.2f' % r2_score(Y_test, y_pred))

rms = sqrt(mean_squared_error(Y_test, y_pred))
# print("RMSE: ", rms)
print("rmse,", rms)

mae = mean_absolute_error(Y_test,y_pred)
# print("MAE: ", mae)
