#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[3]:


# Load the datasets
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')


# In[4]:


# Preprocessing
# Convert 'trans_date' to datetime format in both datasets
train_df['trans_date'] = pd.to_datetime(train_df['trans_date'])
test_df['trans_date'] = pd.to_datetime(test_df['trans_date'])


# In[5]:


label_encoder = LabelEncoder()
train_df['category_encoded'] = label_encoder.fit_transform(train_df['category'])
test_df['category_encoded'] = label_encoder.transform(test_df['category'])


# In[6]:


# Features and target for training and test sets
X_train = train_df[['amount', 'category_encoded']]
y_train = train_df['is_fraud']

X_test = test_df[['amount', 'category_encoded']]
y_test = test_df['is_fraud']


# In[7]:


# Model: Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)


# In[8]:


# Predictions
y_pred = rf_classifier.predict(X_test)


# In[9]:


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)


# In[10]:


# Output results
print(f"Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)


# In[ ]:




