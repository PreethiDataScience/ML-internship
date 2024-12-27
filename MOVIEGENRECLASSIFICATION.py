#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
train_data = pd.read_csv(r"C:\Users\ASUS2023\Desktop\traincopy1.txt",sep=' ::: ', header=None, engine='python')
train_data = pd.read_csv(r"C:\Users\ASUS2023\Desktop\traincopy1.txt",sep=' ::: ', header=None, engine='python')
test_data = pd.read_csv(r"C:\Users\ASUS2023\Desktop\testcopy1.txt",sep=' ::: ', header=None, engine='python')
train_data.columns=['SI.NO','MOVIE','MOVIETYPE','SUMMARY']
test_data.columns=['SI.NO','MOVIE','SUMMARY']
train_data.head()
test_data.head()  


# In[2]:


train_data.info()
test_data.info()
train_data.describe()
test_data.describe()
train_data.isnull().sum()
test_data.isnull().sum()
train_data.count()
test_data.count()


# In[3]:


train_data.iloc[0:3]
train_data.loc[0]
train_data.shape
test_data.shape
sns.countplot(x='MOVIETYPE', data=train_data)
plt.xlabel('Movie Category')
plt.ylabel('Count') 
plt.title('Movie Genre Plot')
plt.xticks(rotation=90);
plt.show()


# In[4]:


sns.displot(train_data.MOVIETYPE, kde =True, color = "black")
plt.xticks(rotation=98);
plt.figure(figsize = (14,10))
count1=train_data.MOVIETYPE.value_counts()
sns.barplot(x=count1,y=count1.index,orient='h',color='pink')
plt.xlabel('Count') 
plt.xlabel('movie type') 
plt.title('Movie Genre Plot')
plt.xticks(rotation=90);
plt.show()


# In[17]:


sns.displot(train_data.MOVIETYPE, kde =True, color = "black")
plt.figure(figsize=(15,15))
plt.pie(train_data['MOVIETYPE'].value_counts(),labels=[' drama ', ' thriller ', ' adult ', ' documentary ', ' comedy ',
       ' crime ', ' reality-tv ', ' horror ', ' sport ', ' animation ',
       ' action ', ' fantasy ', ' short ', ' sci-fi ', ' music ',
       ' adventure ', ' talk-show ', ' western ', ' family ', ' mystery ',
       ' history ', ' news ', ' biography ', ' romance ', ' game-show ',
       ' musical ', ' war '],autopct='%0.1f%%')


# In[14]:


# Check if there are any missing values in the 'combined' DataFrame
combined=pd.concat([train_data,test_data],axis=0)
combined.head()
missing_values = combined.isnull().any().any()

# Check if there are any duplicated rows in the 'combined' DataFrame
duplicates = combined.duplicated().any()

# Print results
print("Missing values:", missing_values)
print("Duplicates:", duplicates)


# In[8]:


test_solution = pd.read_csv(r"C:\Users\ASUS2023\Desktop\test_data_solution.txt",sep=' ::: ', header=None, engine='python')
test_solution.columns=['SI.NO','MOVIE','MOVIETYPE','SUMMARY']
test_solution.head()


# In[11]:


tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1, 2))
label_encoder = LabelEncoder()
train_data['MOVIETYPE'] = label_encoder.fit_transform(train_data['MOVIETYPE'])
test_solution['MOVIETYPE'] = label_encoder.fit_transform(test_solution['MOVIETYPE'])
x_train = tfidf.fit_transform(train_data['SUMMARY'])
x_test = tfidf.transform(test_data['SUMMARY'])
y_train = train_data['MOVIETYPE']
y_test = test_solution['MOVIETYPE']
feature_names = tfidf.get_feature_names_out()
print("Feature Names:", feature_names)
print("X_train shape:", x_train.shape)
print("Y_train length:", len(y_train))
print("y_train_split length:", len(y_train))
print("y_test_split length:", len(y_test))
x_train, x_test, y_train ,y_test = train_test_split(x_train, y_train, test_size=0.25, random_state=4)
print("X_train_split shape:", x_train.shape)
print("X_test_split shape:", x_test.shape)
print("y_train_split length:", len(y_train))
print("y_test_split length:", len(y_test))
model = MultinomialNB(alpha=0.5)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
logistic_regression_model=LogisticRegression()
logistic_regression_model.fit(x_train,y_train)
lr_predict=logistic_regression_model.predict(x_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, lr_predict)
mae = mean_absolute_error(y_test, lr_predict)
r2 = r2_score(y_test, lr_predict)


# In[12]:


print("Linear Regression Performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R2 Score: {r2}")
print("accuracy score:",accuracy_score(y_test,lr_predict))      


# In[ ]:




