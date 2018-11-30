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


df = pd.read_csv("./dataset.orig/carinsurance/carInsurance_train.csv")
df_test = pd.read_csv("./dataset.orig/carinsurance/carInsurance_test.csv")
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
plt.clf()
sns.boxplot(x='Balance',data=df_test,palette='hls')

# In[9]:


df.Balance.max()
df_test.Balance.max()


# In[10]:

# removing outliers
df[df['Balance'] == 98417]

df_test[df_test['Balance'] == 41630]

# In[11]:


df_new = df.drop(df.index[1742])
df_test_new = df_test.drop(df.index[602])


# In[12]:


df_new.isnull().sum()
df_test_new.isnull().sum()


# In[13]:


df_new['Job'] = df_new['Job'].fillna(method ='pad')
df_new['Education'] = df_new['Education'].fillna(method ='pad')

df_test_new['Job'] = df_test_new['Job'].fillna(method ='pad')
df_test_new['Education'] = df_test_new['Education'].fillna(method ='pad')


# In[14]:


# Using none to fill Nan values in Communication and Outcome fields
df_new['Communication'] = df_new['Communication'].fillna('none')
df_new['Outcome'] = df_new['Outcome'].fillna('none')
df_new['CallDurationMinutes'] = (pd.to_datetime(df_new['CallEnd']) - pd.to_datetime(df_new['CallStart'])).astype('timedelta64[m]')

df_test_new['Communication'] = df_test_new['Communication'].fillna('none')
df_test_new['Outcome'] = df_test_new['Outcome'].fillna('none')
df_test_new['CallDurationMinutes'] = (pd.to_datetime(df_test_new['CallEnd']) - pd.to_datetime(df_test_new['CallStart'])).astype('timedelta64[m]')


# In[15]:


df_new.isnull().sum()
df_test_new.isnull().sum()


# In[16]:


df_new.head()
df_test_new.head()


# In[18]:


df_new.to_csv("./dataset.preprocessed/carinsurance/cleaned_train.csv")
df_test_new.to_csv("./dataset.preprocessed/carinsurance/cleaned_test.csv")


# In[ ]:




