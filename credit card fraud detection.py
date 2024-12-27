#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,classification_report



import warnings

warnings.filterwarnings('ignore')


# In[6]:


# Load the dataset

train_data = pd.read_csv('fraudTraincopy2.csv')

test_data = pd.read_csv('fraudTestcopy1.csv')


# In[8]:


print(train_data)


# In[9]:


print(test_data)


# In[10]:


train_data.info()


# In[11]:


train_data.isnull().sum()


# In[12]:


train_data.head(3)


# In[13]:


test_data.info()


# In[14]:


# Check for null values

test_data.isnull().sum()


# In[15]:


test_data.head(3)


# In[16]:


data = pd.concat([train_data,test_data])


# In[17]:


data.shape


# In[18]:


# Let us check how each feature is correlated with the target feature



corr_result = {}



for col in data.columns:

    if data[col].dtype != 'object' and col != 'is_fraud':

        corr = data[col].corr(data['is_fraud'])

        corr_result[col] = corr

        

corr_result


# In[19]:


# Consider only columns necessary for prediction

data.drop(columns=['Unnamed: 0','trans_date_trans_time','first','last','gender','street','job','dob','trans_num'],inplace=True)


# In[20]:


data.info()


# In[21]:


# Check class distribution

data['is_fraud'].value_counts()


# In[24]:


plt.bar(data['is_fraud'].unique(),data['is_fraud'].value_counts(),width = 0.2)

plt.xlabel('Legitimate/Fraud')

plt.ylabel('No of transactions')

plt.show()


# In[29]:


plt.scatter(data['is_fraud'].unique(),data['is_fraud'].value_counts())

plt.xlabel('Legitimate/Fraud')

plt.ylabel('No of transactions')

plt.show()


# In[40]:


sns.violinplot(x=data['is_fraud'],color="cyan")

plt.xlabel('Legitimate/Fraud')

plt.show()


# In[41]:


# Separate fraud & legitimate transactions

legitimate = data[data['is_fraud'] == 0]

fraud = data[data['is_fraud'] == 1]


# In[42]:


# We'll consider only a sample(same number of transactions as fraud) of legitimate transactions

legitimate = legitimate.sample(n = len(fraud))

legitimate.shape


# In[43]:


# We have same no of legitimate & fraud transactions

fraud.shape


# In[44]:


# Combine the data

data = pd.concat([legitimate,fraud])


# In[45]:


# Check class distribution

plt.bar(data['is_fraud'].unique(),data['is_fraud'].value_counts(),width = 0.2)

plt.xlabel('Legitimate/Fraud')

plt.ylabel('No of transactions')

plt.show()


# In[46]:


# Encoding categorical data

le = LabelEncoder()

data['merchant'] = le.fit_transform(data['merchant'])

data['category'] = le.fit_transform(data['category'])

data['city'] = le.fit_transform(data['city'])

data['state'] = le.fit_transform(data['state'])


# In[47]:


# All the features are numerical

data.info()


# In[48]:


# Separating the target feature

x_data = data.iloc[:,:-1].values

y_data = data.iloc[:,-1].values


# In[49]:


x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.3,random_state=42,shuffle=True)


# In[50]:


log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)


# In[51]:


log_pred = log_reg.predict(x_train)

print("Accuracy score: ",round(accuracy_score(y_train,log_pred),3))

print("Classification report:\n",classification_report(y_train,log_pred))


# In[52]:


dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)


# In[53]:


dt_pred = dt.predict(x_train)

print("Accuracy score: ",round(accuracy_score(y_train,dt_pred),3))

print("Classification report:\n",classification_report(y_train,dt_pred))


# In[54]:


rfc = RandomForestClassifier(n_estimators = 70)

rfc.fit(x_train,y_train)


# In[55]:


rfc_pred = rfc.predict(x_train)

print("Accuracy score: ",round(accuracy_score(y_train,rfc_pred),3))

print("Classification report:\n",classification_report(y_train,rfc_pred))


# In[56]:


test_pred = log_reg.predict(x_test)

print("Accuracy score: ",round(accuracy_score(y_test,test_pred),3))

print("Classification report:\n",classification_report(y_test,test_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




