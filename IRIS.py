#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IRIS flower classification
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


# In[2]:


df=pd.read_csv('Iris.csv')


# In[3]:


df.head(5)


# In[4]:


df=df.drop(columns=['Id'])
df.head(5)


# In[5]:


df.describe()


# In[7]:


df.info()


# In[8]:


#number of unique values
df['Species'].value_counts()


# In[8]:


df.isnull().sum()


# In[9]:


#distribution in histogram
df['SepalLengthCm'].hist()


# In[10]:


df['SepalWidthCm'].hist()


# In[11]:


df['PetalLengthCm'].hist()


# In[12]:


df['PetalWidthCm'].hist()


# In[13]:


# visualise
colors=['red','yellow','green']
species=['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[14]:


for i in range(3):
   x=df[df['Species']==species[i]]
   plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[15]:


for i in range(3):
   x=df[df['Species']==species[i]]
   plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[16]:


for i in range(3):
   x=df[df['Species']==species[i]]
   plt.scatter(x['SepalLengthCm'],x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[17]:


for i in range(3):
   x=df[df['Species']==species[i]]
   plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# In[18]:


#coorelation matrix 
df.corr()


# In[19]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(5,6))
sns.heatmap(corr,annot=True, ax=ax)


# In[20]:


#label encoder
le=LabelEncoder()
df['Species']=le.fit_transform(df['Species'])
df.head()


# In[21]:


#model training
from sklearn.model_selection import train_test_split
X=df.drop(columns=['Species'])
Y=df['Species']
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.30)


# In[22]:


from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(x_train,y_train)


# In[23]:


print("Accracy:",model.score(x_test, y_test)*100)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(x_train, y_train)


# In[25]:


print("accuracy:", model.score(x_test, y_test)*100)


# In[26]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# In[27]:


print("accuracy:", model.score(x_test, y_test)*100)


# In[ ]:





# In[ ]:




