#!/usr/bin/env python
# coding: utf-8

# # importing the libraries
# 

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


df=pd.read_csv("churncopy.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


list(df.columns)


# In[7]:


df.isnull().sum()


# In[ ]:


#Since there are no null values, the data does not require cleaning.

Had there been null values, we would have either deleted the rows with null values or used the fillna methods.


# In[8]:


df.drop(columns=["RowNumber","CustomerId","Surname"],inplace = True)


# In[10]:


plt.figure(figsize=(15, 6))
sns.histplot(df['CreditScore'], bins=30, kde=True, color='purple')
plt.title('Distribution of Credit Scores')
plt.xlabel('Credit Score')
plt.ylabel('Frequency')
plt.show()


# In[ ]:


#The plot shows that the distribution of credit score is slightly skewed(w.r.t. a normal distribution) with more points concentrated in the range <700


# In[12]:


plt.figure(figsize=(15,6))
sns.histplot(data =df,x="Age",kde = True,bins = 30,color = "cyan")
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[14]:


plt.figure(figsize=(15,6))
sns.histplot(data =df,x="Tenure",kde = True,bins = 5,color = "red")
plt.title('Tenure Distribution')
plt.xlabel('Tenure(years)')
plt.ylabel('Frequency')
plt.show()


# In[15]:


plt.figure(figsize=(10, 6))
sns.histplot(df['Balance'], bins=35, kde=True, color='brown')
plt.title('Balance Distribution')
plt.xlabel('Balance')
plt.ylabel('Frequency')
plt.show()


# In[16]:


plt.figure(figsize=(10, 6))
sns.histplot(df['EstimatedSalary'], bins=30, kde=True, color='coral')
plt.title('Estimated Salary Distribution')
plt.xlabel('Estimated Salary')
plt.ylabel('Frequency')
plt.show()


# In[18]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Geography', palette='Set3')
plt.title('Geographical Distribution')
plt.xlabel('Geography')
plt.ylabel('Count')
plt.show()


# In[19]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Gender', palette='coolwarm')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


# In[21]:


plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='NumOfProducts', palette='Set2')
plt.title('Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Count')
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='HasCrCard', palette='Set2')
plt.title('Credit Card Holders')
plt.xlabel('Has Credit Card')
plt.ylabel('Count')
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Exited', palette='Set3')
plt.title('Churn Distribution')
plt.xlabel('Exited')
plt.ylabel('Count')
plt.show()


# In[29]:


plt.figure(figsize=(10, 6))
plt.scatter(data=df, x='Age', y='Balance',color='red')
plt.title('Age vs. Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()


# In[31]:


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Age', y='Balance', hue='Exited')
plt.title('Age vs. Balance')
plt.xlabel('Age')
plt.ylabel('Balance')
plt.show()


# In[33]:


df = pd.get_dummies(df, columns=['Geography'],dtype=int,drop_first=True)
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
#Feature Engineering

df['IsSenior'] = df['Age'].apply(lambda x: 1 if x >= 60 else 0)
df['IsActive_by_CreditCard'] = df['HasCrCard'] * df['IsActiveMember']


# In[34]:


X_prev = df.drop("Exited", axis=1)
y_prev = df['Exited']
X_prev_train, X_prev_test, y_prev_train, y_prev_test = train_test_split(X_prev, y_prev, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_prev_train)
X_test=sc.transform(X_prev_test)
model_1 = LogisticRegression()
model_1.fit(X_prev_train, y_prev_train)
pred_1 = model_1.predict(X_prev_test)
accuracy_score_1_prev=accuracy_score(y_prev_test, pred_1)
print(f'Logistic Regression Model Accuracy: {accuracy_score_1_prev:.4f}')


# In[36]:


model_2 = KNeighborsClassifier(n_neighbors=2)
model_2.fit(X_prev_train,y_prev_train)
pred_knn = model_2.predict(X_prev_test)
accuracy_score_2_prev = accuracy_score(y_prev_test, pred_knn)
print(f'KNN Model Accuracy: {accuracy_score_2_prev:.2f}')


# In[37]:


model_3 = BernoulliNB()
model_3.fit(X_prev_train, y_prev_train)
pred_3 = model_3.predict(X_prev_test)
accuracy_score_3_prev=accuracy_score(y_prev_test, pred_3)
print(f'BernoulliNB Model Accuracy: {accuracy_score_3_prev:.3f}')


# In[38]:


model_4 = RandomForestClassifier()
model_4.fit(X_prev_train, y_prev_train)
pred_4= model_4.predict(X_prev_test)
accuracy_score_4_prev=accuracy_score(y_prev_test, pred_4)
print(f'RandomForestClassifier Model Accuracy: {accuracy_score_4_prev:.3f}')


# In[41]:


corr_data = df.corr()
plt.figure(figsize=(10, 5))
sns.heatmap(data=corr_data, annot=True, cmap='coolwarm')
plt.title("Correlation with Label")
plt.show()


# In[43]:


X = df.drop("Exited", axis=1)
y = df['Exited']
from sklearn.feature_selection import chi2, f_classif, SelectKBest

# Apply chi-square test
chi2_selector = SelectKBest(score_func=chi2, k='all')
chi2_selector.fit(X, y)

# Get scores and p-values
chi2_scores = chi2_selector.scores_
p_values = chi2_selector.pvalues_

# Create a DataFrame to display feature scores
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'Chi2 Score': chi2_scores,
    'P-Value': p_values
}).sort_values(by='Chi2 Score', ascending=True)

print(pd.DataFrame(feature_scores))


# In[44]:


X=df.drop(['CreditScore','IsActive_by_CreditCard'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_1 = LogisticRegression()
model_1.fit(X_train, y_train)
pred_1 = model_1.predict(X_test)
accuracy_score_1=accuracy_score(y_test, pred_1)
print(f'Logistic Regression Model Accuracy: {accuracy_score_1:.4f}')
print("Classification Report:")
print(classification_report(y_test, pred_1))


# In[45]:


model_knn = KNeighborsClassifier(n_neighbors=2)
model_knn.fit(X_train,y_train)
pred_knn = model_knn.predict(X_test)
accuracy_score_2 = accuracy_score(y_test, pred_knn)
print(f'KNN Model Accuracy: {accuracy_score_2:.4f}')
print("Classification Report:")
print(classification_report(y_test, pred_knn))


# In[46]:


model_3 = BernoulliNB()
model_3.fit(X_train, y_train)
pred_3 = model_3.predict(X_test)
accuracy_score_3=accuracy_score(y_test, pred_3)
print(f'BernoulliNB Model Accuracy: {accuracy_score_3:.4f}')
print("Classification Report:")
print(classification_report(y_test, pred_3))


# In[47]:


model_4 = RandomForestClassifier()
model_4.fit(X_train, y_train)
pred_4= model_4.predict(X_test)
accuracy_score_4=accuracy_score(y_test, pred_4)
print(f'RandomForestClassifier Model Accuracy: {accuracy_score_4:.4f}')
print("Classification Report:")
print(classification_report(y_test, pred_4))


# In[48]:


accuracy_scores = {
    'Logistic Regression': accuracy_score_1,
    'KNN': accuracy_score_2,
    'BernoulliNB': accuracy_score_3,
    'Random Forest': accuracy_score_4
}

algos = list(accuracy_scores.keys())
scores = list(accuracy_scores.values())

plt.figure(figsize=(10, 6))
plt.bar(algos, scores, color=['pink', 'teal', 'orange', 'green'])
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.title("Accuracy Scores for Different Algorithms")
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for accuracy
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




