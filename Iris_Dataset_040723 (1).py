#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


iris = pd.read_csv("C:\\Users\\PCC\\Downloads\\archive.csv\\IRIS.csv")


# In[4]:


iris


# In[5]:


iris.columns


# In[6]:


iris.isna()


# In[7]:


iris.isnull()


# In[8]:


iris.isna().sum()


# In[9]:


iris.dropna()


# # Encoding Categorical Data

# In[10]:


# 1 Label Encoder
# 2 One hot encoder


# In[11]:


from sklearn import preprocessing


# In[12]:


le = preprocessing.LabelEncoder()


# In[13]:


data = pd.DataFrame({'Animals':['Cat','Dog','Horse']})


# In[14]:


data


# In[15]:


le.fit(data)


# In[16]:


encoded_data = le.transform(data)


# In[17]:


encoded_data


# In[18]:


# one hot encoding

ohe = preprocessing.OneHotEncoder()


# In[19]:


ohe.fit(data)


# In[20]:


encoded_ohe_data = ohe.transform(data)


# In[21]:


encoded_ohe_data.toarray()


# In[22]:


# Manual
# Input - x
# output - y


# In[23]:


x = iris.iloc[:,:4]


# In[24]:


x


# In[25]:


y = iris.iloc[:,4]


# In[26]:


y


# # Training and Testing 
# 

# In[27]:


x_train = x.iloc[:100]


# In[28]:


x_test = x.iloc[100:]


# In[29]:


x_train


# In[30]:


x_test


# In[31]:


y_train = y[:100]


# In[32]:


y_test = y[100:]


# In[33]:


y_train


# In[34]:


y_test


# In[35]:


# 2nd 

# Sklearn


# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 30 , random_state=1)


# In[38]:


x_train


# In[39]:


x_test


# In[40]:


y_train


# In[41]:


y_test


# In[ ]:




