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

warnings.filterwarnings("ignore")

df = pd.read_csv("../dataset.preprocessed/carinsurance/cleaned.csv")
print("Number of data points:",df.shape[0])
X = df[['Balance']]
y = df["CarLoan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

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
print("Accuracy on test set: %0.3f"%(accuracy_score(y_test, y_pred)))
print("F1-Score on test set: %0.2f"%(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)))
