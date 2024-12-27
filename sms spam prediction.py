#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


data=pd.read_csv("spamcopy.csv",encoding='latin1')
data.head()


# In[12]:


data.info()


# In[13]:


data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=False)


# In[14]:


data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], inplace=True)
data.head(2)


# In[15]:


data.columns=["target", "sms"]


# In[16]:


data.head(2)


# In[18]:


data["target"]=data["target"].replace("ham", 0)
data["target"]=data["target"].replace("spam", 1)
data.head(2)


# In[20]:


data.isnull().sum()


# In[21]:


data.duplicated().sum()


# In[22]:


len(data)


# In[24]:


data=data.drop_duplicates()
data.shape


# In[25]:


data["target"].value_counts()


# In[41]:


plt.pie(data["target"].value_counts(), labels=["Ham", "Spam"], autopct="%.3f")
plt.show()


# In[28]:


sns.boxplot(data["target"].value_counts())
plt.show()


# In[79]:


sns.histplot(data["target"],bins=30,color="red")
plt.show()


# In[40]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
text=str(data[data["target"]==1]["sms"])
wordcloud = WordCloud().generate(text)

plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[42]:


import re
def remove_urls(text):
    return re.sub(r'http\S+', '', text)


# In[43]:


def remove_punctuations(text):
    text=re.sub(r"[^A-Za-z0-9\s]", "", text)
    return text


# In[44]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def remove_stopword(text):
    stop_words= stopwords.words('english')
    temp_text = word_tokenize(text)
    for word in temp_text:
        if word in stop_words:
            text=text.replace(word, "")

    return text


# In[45]:


from nltk.stem import PorterStemmer
def Stemming(text):
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_words = []
    for token in tokens:
        stemmed_token = ps.stem(token)
        stemmed_words.append(stemmed_token)
    return ' '.join(stemmed_words)


# In[46]:


def transform(text):
    text=text.lower()
    text=remove_urls(text)
    text=remove_punctuations(text)
    text=remove_stopword(text)
    text=Stemming(text)
    return text


# In[ ]:


data["transformed"]=data["sms"].apply(transform)


# In[52]:


data.head(2)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf= TfidfVectorizer()
X=tf.fit_transform(data["transformed"]).toarray()


# In[56]:


Y=data["target"]


# In[65]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load a dataset
data = load_iris()
X = data.data  # Features
y = data.target  # Labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can use X_train in the next steps, e.g., training a model
print(X_train)
print(y_test)


# In[67]:


from sklearn.naive_bayes import BernoulliNB
bnb_model=BernoulliNB()
bnb_model.fit(X_train, y_train)


# In[68]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[71]:


from sklearn.metrics import f1_score

# Example of a multiclass classification problem
y_true = [0, 1, 2, 2, 0, 1]
y_pred = [0, 0, 2, 2, 0, 2]

# If you use average='binary', you will get the error
# f1_score(y_true, y_pred, average='binary')  # This will raise the ValueError

# Fix the error by choosing an appropriate average value for multiclass
f1_score_micro = f1_score(y_true, y_pred, average='micro')  # Use 'micro' for multiclass
f1_score_macro = f1_score(y_true, y_pred, average='macro')  # Use 'macro' for multiclass
f1_score_weighted = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass

print(f"Micro-average F1 score: {f1_score_micro}")
print(f"Macro-average F1 score: {f1_score_macro}")
print(f"Weighted-average F1 score: {f1_score_weighted}")


# In[73]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_pred, labels=[1, 0])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Spam", "Ham"], yticklabels=["SPam", "Ham"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix For Spam Detection")


# In[75]:


from sklearn.linear_model import LogisticRegression
lr_model= LogisticRegression()
lr_model.fit(X_train, y_train)


# In[76]:


from sklearn.model_selection import cross_val_score
y_train1=lr_model.predict(X_train)
y_test1= lr_model.predict(X_test)
print("Accuracy X_Train", accuracy_score(y_train, y_train1))
print("Accuracy X_Test", accuracy_score(y_test, y_test1))
print("Cross_Validation", cross_val_score(lr_model, X_train, y_train, cv=5).mean())


# In[78]:


cm = confusion_matrix(y_test, Y_pred, labels=[1, 0])
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Spam", "Ham"], yticklabels=["Spam", "Ham"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


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




