#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, accuracy_score

import joblib


# In[2]:


df = pd.read_csv("D:\CC\SEM 3\COMP 258 Neural Networks\Assignments\Group Project\Student data.csv")


# ### Preprocessing

# In[3]:


df = df.apply(pd.to_numeric, errors='coerce', downcast='float')
df.replace('?', np.nan, inplace=True)


# In[4]:


correlation_matrix = df.corr()
print(correlation_matrix['FirstYearPersistence'].sort_values(ascending=False))


# ### Dropping cloumns which are not relevant for prediction
# 

# In[5]:


columns_to_drop = ['School', 'Gender', 'Coop', 'English Grade', 'Age Group', 'Previous Education']
df.drop(columns_to_drop, axis=1, inplace=True)


# ### Remaining columns

# In[6]:


print(df.columns)


# In[7]:


X = df.drop(['FirstYearPersistence'], axis=1)
y = df['FirstYearPersistence']


# ### Tarin and Test Split

# In[8]:


StratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=10) #splt the data 20% test
for train_index, test_index in StratifiedShuffleSplit.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]


# ### Preprocessing the data

# In[9]:


def preprocessing_pipeline(data):
    category_columns = list(data.select_dtypes(include=object).columns)

    transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean', fill_value=None)),
        ('scaler', StandardScaler())
    ])
    
    return transformer


# In[10]:


def create_model(layers, learning_rate=0.001):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes, input_dim=X_train.shape[1]))
            model.add(Activation('relu'))
        else:
            model.add(Dense(nodes))
            model.add(Activation('relu'))
            
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    
    optimizer = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model)


# In[11]:


new_pre_processing_pipeline = preprocessing_pipeline(X_train)

pipeline = Pipeline(steps=
                        [
                            ('new_pipeline', new_pre_processing_pipeline),
                            ('classifier', model)
                        ])


# In[12]:


param_grid = {
    'classifier__layers': [[9], [9, 6, 4], [9, 6, 4, 2]],
    'classifier__epochs': [10, 100],
    'classifier__batch_size': [32, 64, 128]
}

grid_search = GridSearchCV(estimator=pipeline,
                               param_grid=param_grid,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=10,
                               cv=5)


# In[ ]:


grid_search.fit(X_train, y_train)


# In[ ]:


best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_


# In[ ]:


print(f"Best parameters {best_params}")


# In[ ]:


y_pred = grid_search.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)
cm


# In[ ]:


accuracy_score(y_test, y_pred)


# ### Dumping the best model

# In[ ]:


joblib.dump(best_estimator, 'best_model.pkl')


# ### Importing the model

# In[ ]:


new_model = joblib.load('best_model.pkl')


# ### Testing the model with random data

# In[ ]:


testset = [2.210, 1375, 1, 2, 2, 1, 68, 20]


# In[ ]:


new_model.predict([testset])


# In[ ]:




