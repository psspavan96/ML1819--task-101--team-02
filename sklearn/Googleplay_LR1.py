#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from subprocess import check_output
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import os
import gc

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup


# In[2]:


df = pd.read_csv("/Users/pavanpss/Downloads/google-play-store-apps/googleplaystore.csv")
print("Number of data points:",df.shape[0])


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df = df[df["Rating"]<=5]


# In[6]:


df.Category.unique()


# In[7]:


CategoryString = df["Category"]
categoryVal = df["Category"].unique()
categoryValCount = len(categoryVal)
category_dict = {}
for i in range(0,categoryValCount):
    category_dict[categoryVal[i]] = i
df["Category"] = df["Category"].map(category_dict).astype(int)


# In[8]:


df["Genres"].unique()


# In[9]:


genresString = df["Genres"]
genresVal = df["Genres"].unique()
genresValCount = len(genresVal)
genres_dict = {}
for i in range(0,genresValCount):
    genres_dict[genresVal[i]] = i
df["Genres"] = df["Genres"].map(genres_dict).astype(int)


# In[10]:


df['Content Rating'].unique()


# In[11]:


df['Content Rating'] = df['Content Rating'].map({'Everyone':0,'Teen':1,'Everyone 10+':2,'Mature 17+':3,'Adults only 18+':4}).astype(float)


# In[12]:


df['Reviews'] = [ float(i.split('M')[0]) if 'M'in i  else float(i) for i in df['Reviews']]


# In[13]:


df["Size"] = [ float(i.split('M')[0]) if 'M' in i else float(0) for i in df["Size"]  ]


# In[14]:


df['Price'] = [ float(i.split('$')[1]) if '$' in i else float(0) for i in df['Price'] ]


# In[15]:


df.Installs.unique()


# In[16]:


df["Installs"] = [ float(i.replace('+','').replace(',', '')) if '+' in i or ',' in i else float(0) for i in df["Installs"] ]


# In[17]:


df.drop(["Last Updated","Current Ver","Android Ver","App","Type"],axis=1,inplace=True)


# In[18]:


df["Rating"] = df.groupby("Category")["Rating"].transform(lambda x: x.fillna(x.mean()))
df["Content Rating"] = df[["Content Rating"]].fillna(method="ffill")


# In[19]:


df.info()


# In[20]:


df.shape


# In[21]:


df.head()


# In[22]:


del df["Content Rating"]
df


# In[24]:



del df["Installs"]
del df["Price"]
del df["Genres"]
df


# In[25]:


from sklearn.model_selection import train_test_split
X = df.drop('Rating', axis = 1)
Y = df['Rating']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[26]:


from sklearn.linear_model import LinearRegression
from matplotlib import *

lm = LinearRegression()
lm.fit(X_train, Y_train)

lm.score(X_test, Y_test)


# In[27]:


lm.coef_ 


# In[28]:


lm.intercept_


# In[29]:


lm.predict(X_test)


# In[31]:


y_pred = lm.predict(X_test) 
plt.plot(Y_test, y_pred, '.')
plt.xlabel('Measured')
plt.ylabel('Predicted')
# plot a line, a perfit predict would all fall on this line
x = np.linspace(0, 10)
y = x
plt.plot(x, y)
plt.show()


# In[34]:


from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"% mean_squared_error(Y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, y_pred))


# In[35]:


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test, y_pred))
print(rms)


# In[36]:


from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(Y_test,y_pred)
print(mae)


# In[ ]:




