#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm,tree


# In[2]:


df = pd.read_csv("/Users/pavanpss/Desktop/cleandata.csv")
print("Number of data points:",df.shape[0])


# In[3]:


del df['Age']
del df['Marital']
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
del df['CallDurationMinutes']


# In[4]:


df


# In[5]:


del df['NoOfContacts']


# In[7]:


del df['Unnamed: 0']
del df['Job']
del df['LastContactDay']
df


# In[8]:


from sklearn.model_selection import train_test_split
X = df.drop(["CarLoan"],axis=1)
y = df["CarLoan"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[9]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[10]:


from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

clf = SVC()

param_grid = {'gamma':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],
             'C':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]} #params we need to try on classifier
gscv = GridSearchCV(clf,param_grid,verbose=1,n_jobs=-1,scoring='accuracy', cv=10)
gscv.fit(X_train,y_train)
print("Best HyperParameter: ",gscv.best_params_)
print("Best Accuracy: %.2f%%"%(gscv.best_score_*100))


# In[11]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sn
clf = SVC(C=0.5,gamma=0.5)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))
print("F1-Score on test set: %0.2f"%(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)))


# In[12]:


SVM = svm.SVC()
SVM.fit(X_train, y_train)
print ("SVM Accuracy is %2.2f" % accuracy_score(y_test, SVM.predict(X_test)))
score_svm = cross_val_score(SVM, X, y, cv=10).mean()
print("Cross Validation Score = %2.2f" % score_svm)
y_pred = SVM.predict(X_test)
print(classification_report(y_test,y_pred))


# In[13]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, y_pred))
print(rms)
mse = mean_squared_error(y_test, y_pred)
print(mse)


# In[ ]:




