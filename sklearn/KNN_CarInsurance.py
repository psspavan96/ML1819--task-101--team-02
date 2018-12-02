#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm,tree


# In[2]:


df = pd.read_csv("../dataset.preprocessed/carinsurance/cleaned.csv")
print("Number of data points:",df.shape[0])


# In[3]:


del df['Education']
del df['Default']
del df['HHInsurance']
del df['Communication']
del df['LastContactMonth']
del df['DaysPassed']
del df['PrevAttempts']
del df['Outcome']
del df['CallStart']
del df['CallEnd']


# In[4]:


df


# In[5]:


del df['Unnamed: 0']
del df['CarLoan']
del df['Job']
del df['LastContactDay']


# In[6]:


df


# In[7]:

import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
X = df[['Age', 'Balance', 'NoOfContacts', 'CallDurationMinutes']]
y = df[['Marital']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[8]:


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# In[10]:


# from sklearn.model_selection import GridSearchCV
# knn = KNeighborsClassifier(algorithm='auto')
# param_grid = {'n_neighbors':np.arange(1,100,2)} #params we need to try on classifier
# gscv = GridSearchCV(knn,param_grid,verbose=1,cv=10)
# gscv.fit(X_train,y_train)
# print("Best HyperParameter: ",gscv.best_params_)
# print("Best Accuracy: %.2f%%"%(gscv.best_score_*100))


# In[11]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn

knen = KNeighborsClassifier(n_neighbors=4,algorithm='auto')
knen.fit(X_train,y_train)
y_pred = knen.predict(X_test)
# print(y_pred)
print("Accuracy on test set: %0.2f%%"%(accuracy_score(y_test, y_pred)*100))


# In[12]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
print("RSME: ", rms)
mse = mean_squared_error(y_test, y_pred)
print("MSE: ", mse)


# In[ ]:




