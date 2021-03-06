#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split,cross_val_score,KFold,cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score,confusion_matrix,precision_recall_curve,roc_curve
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors  import KNeighborsClassifier
from sklearn import svm,tree


# In[2]:


df_train = pd.read_csv("./dataset.orig/carinsurance/carInsurance_train.csv")
df_test = pd.read_csv("./dataset.orig/carinsurance/carInsurance_test.csv")

df = pd.concat([df_train, df_test], ignore_index=True)
df = df.drop('CarInsurance', axis=1)
df = df.drop('Id', axis=1)
print("Number of data points:",df.shape[0])


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


# Statistics of categorical features
df.describe(include=['O'])


# In[8]:


sns.boxplot(x='Balance',data=df,palette='hls')

# In[9]:


df.Balance.max()


# In[10]:

# removing outliers
df[df['Balance'] == 98417]

# In[11]:


df_new = df.drop(df.index[1742])


# In[12]:


df_new.isnull().sum()


# In[13]:


df_new['Job'] = df_new['Job'].fillna(method ='pad')
df_new['Education'] = df_new['Education'].fillna(method ='pad')

# In[14]:


# Using none to fill Nan values in Communication and Outcome fields
df_new['Communication'] = df_new['Communication'].fillna('none')
df_new['Outcome'] = df_new['Outcome'].fillna('none')
df_new['CallDurationMinutes'] = (pd.to_datetime(df_new['CallEnd']) - pd.to_datetime(df_new['CallStart'])).astype('timedelta64[m]')



# In[15]:


df_new.isnull().sum()


# In[16]:


df_new.head()


# In[18]:

df_new['Job'] = df_new['Job'].astype('category')
df_new['Job'] = df_new['Job'].cat.codes

df_new['Marital'] = df_new['Marital'].astype('category')
df_new['Marital'] = df_new['Marital'].cat.codes

df_new['Education'] = df_new['Education'].astype('category')
df_new['Education'] = df_new['Education'].cat.codes

df_new['Communication'] = df_new['Communication'].astype('category')
df_new['Communication'] = df_new['Communication'].cat.codes

df_new['LastContactMonth'] = df_new['LastContactMonth'].astype('category')
df_new['LastContactMonth'] = df_new['LastContactMonth'].cat.codes

df_new['Outcome'] = df_new['Outcome'].astype('category')
df_new['Outcome'] = df_new['Outcome'].cat.codes


df_new.to_csv("./dataset.preprocessed/carinsurance/cleaned.csv")

# In[ ]:




