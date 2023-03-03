#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Loading all the libraries
import pandas as pd
import numpy as np
from sklearn import svm, datasets
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#loading the dataset
BC_Data = datasets.load_breast_cancer()


# In[3]:


#Exploring the dataset
print(BC_Data.DESCR)


# In[4]:


#Parititioning the data
X_train, X_test, y_train, y_test = train_test_split(BC_Data.data,
BC_Data.target, random_state=0)


# In[5]:


#Fitting the model by using linear kernel
C= 1.0
svm= SVC(kernel="linear",C=C)
svm.fit(X_train, y_train)
print('Accuracy-train dataset: {:.3f}'.format(svm.score(X_train,y_train)))
print('Accuracy- test dataset: {:.3f}'.format(svm.score(X_test,y_test)))


# In[6]:


# fitting the model by using rbf kernel
svm= SVC(kernel="rbf",C=C)
svm.fit(X_train, y_train)
print('Accuracy-train dataset: {:.3f}'.format(svm.score(X_train,y_train)))
print('Accuracy- test dataset: {:.3f}'.format(svm.score(X_test,y_test)))


# It's an overfitting case as accuracy on testset is very low as compared to training data. We will go ahead and normalize it

# In[7]:


#Normalizing the data
min_train = X_train.min(axis=0)
range_train = (X_train - min_train).max(axis=0)
X_train_scaled = (X_train - min_train)/range_train
X_test_scaled = (X_test - min_train)/range_train


# Let's fit a model on the scaled data.

# In[8]:


svm= SVC(kernel="rbf",C=C)
svm.fit(X_train_scaled, y_train)
print('Accuracy-train dataset:{:.3f}'.format(svm.score(X_train_scaled,y_train)))
print('Accuracy test dataset:{:.3f}'.format(svm.score(X_test_scaled,y_test)))


# We will get to the optimal hyperparameters now with the help of Grid Search.

# In[9]:


parameters = [{'kernel': ['rbf'],'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],'C': [1, 10, 100, 1000]},{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
clf = GridSearchCV(SVC(decision_function_shape='ovr'), parameters, cv=5)
clf.fit(X_train, y_train)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
print()

