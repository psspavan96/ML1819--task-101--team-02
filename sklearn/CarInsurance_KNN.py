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
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings("ignore")

df = pd.read_csv("../dataset.preprocessed/carinsurance/cleaned.csv")
print("Number of data points:",df.shape[0])

X = df[['Age', 'Balance', 'NoOfContacts', 'CallDurationMinutes']]
y = df[['Marital']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


knen = KNeighborsClassifier(n_neighbors=4,algorithm='auto')
knen.fit(X_train,y_train)
y_pred = knen.predict(X_test)
print("Accuracy on test set: %0.2f%%"%(accuracy_score(y_test, y_pred)*100))

rms = sqrt(mean_squared_error(y_test, y_pred))
print("RSME: ", rms)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)
