#!/usr/bin/env python
# coding: utf-8

# In[95]:


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


# In[96]:


df = pd.read_csv("../dataset.preprocessed/googleplay/cleaned.csv")
# print("Number of data points:",df.shape[0])


# In[97]:


# df.head()


# In[98]:


# df.info()


# In[99]:


del df["Content Rating"]
df


# In[100]:


del df["Rated 4.4 or more"]
del df["Unnamed: 0"]
del df["Price"]
del df["Installs"]
del df["Rating"]
df


# In[101]:


from sklearn.model_selection import train_test_split
X = df[['Reviews', 'Size', 'Genres']]
y = df.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)


# In[102]:


# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


# In[103]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.fit_transform(X_test)


# In[104]:


# lab_enc = preprocessing.LabelEncoder()
# y_train_enc = lab_enc.fit_transform(y_train)
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# import warnings
# warnings.filterwarnings("ignore")
# #using KD-tree
# knn = KNeighborsClassifier(algorithm='auto')
# param_grid = {'n_neighbors':np.arange(1,100,2)} #params we need to try on classifier
# gscv = GridSearchCV(knn,param_grid,verbose=1)
# gscv.fit(X_train_scaled,y_train_enc)
# print("Best HyperParameter: ",gscv.best_params_)
# print("Best Accuracy: %.2f%%"%(gscv.best_score_*100))


# In[105]:


# y_test = lab_enc.fit_transform(y_test)


# In[107]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sn

knen = KNeighborsClassifier(n_neighbors=4,algorithm='auto')
knen.fit(X_train_scaled,y_train)
y_pred = knen.predict(X_test_scaled)
# print(y_pred)
print("Accuracy on test set: %0.2f%%"%(accuracy_score(y_test, y_pred)*100))


# In[108]:


from sklearn.metrics import mean_squared_error
from math import sqrt

mse = mean_squared_error(y_test, y_pred)
print("Mse: ", mse)
rms = sqrt(mse)
print("Rmse: ", rms)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




