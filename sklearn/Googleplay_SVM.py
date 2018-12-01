#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
# get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
# py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# In[81]:


df = pd.read_csv("../dataset.preprocessed/googleplay/cleaned.csv")
# print("Number of data points:",df.shape[0])


# In[82]:


# df.head()


# In[83]:


# df.info()


# In[84]:


del df["Content Rating"]
del df["Unnamed: 0"]
del df["Genres"]
del df["Rating"]
del df["Size"]
del df["Installs"]
del df["Price"]


# In[85]:


del df["Category"]


# In[86]:


# df


# In[88]:


from sklearn.model_selection import train_test_split
X = df[['Reviews']]
y = df["Rated 4.4 or more"]
y[y == -1] =0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)


# In[89]:


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# In[90]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)


# In[91]:


# lab_enc = preprocessing.LabelEncoder()
# y_train_enc = lab_enc.fit_transform(y_train)
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC

# clf = SVC()

# param_grid = {'gamma':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001],
#              'C':[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]} #params we need to try on classifier
# gscv = GridSearchCV(clf,param_grid,verbose=1,n_jobs=-1,scoring='accuracy')
# gscv.fit(X_train_scaled,y_train_enc)
# print("Best HyperParameter: ",gscv.best_params_)
# print("Best Accuracy: %.2f%%"%(gscv.best_score_*100))


# In[92]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import seaborn as sn
# y_test = lab_enc.fit_transform(y_test)
clf = SVC()
clf.fit(X_train_scaled,y_train)
y_pred = clf.predict(X_test_scaled)
# print(y_pred)
print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))
print("F1-Score on test set: %0.2f"%(f1_score(y_test, y_pred, labels=None, pos_label=1, average='macro', sample_weight=None)))


# In[13]:





# In[14]:





# In[15]:





# In[16]:





# In[17]:





# In[18]:





# In[19]:





# In[20]:





# In[21]:





# In[54]:





# In[55]:





# In[56]:





# In[58]:





# In[59]:





# In[64]:





# In[61]:





# In[63]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




