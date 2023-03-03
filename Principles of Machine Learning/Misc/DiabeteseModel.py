#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.externals import joblib


# In[5]:


"""Attributes:

Dataset information:

This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
Several constraints were placed on the selection of these instances from a larger database. 
In particular, all patients here are females at least 21 years old of Pima Indian heritage.


Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction: Diabetes pedigree function

Age: Age (years)

Outcome: Class variable (0 or 1)

"""


# In[7]:


diabetesDF = pd.read_csv('diabetes.csv')
diabetesDF.head()


# In[8]:


diabetesDF.info()


# In[9]:


corr = diabetesDF.corr()
corr


# In[11]:


sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# In[12]:


#Total 768 patients record
#Using 650 data for training
# Using 100 data for testing
#Using 18 data for checking

dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]


# In[30]:


#Separating label and features and converting to numpy array to feed into our model
trainLabel = np.asarray(dfTrain['Outcome'])
trainData = np.asarray(dfTrain.drop('Outcome',1))
testLabel = np.asarray(dfTest['Outcome'])
testData = np.asarray(dfTest.drop('Outcome',1))


# In[31]:


# Normalize the data 
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)

trainData = (trainData - means)/stds
testData = (testData - means)/stds

# means = np.mean(trainData, axis=0)
# stds = np.std(trainData, axis=0)


# In[32]:


#Now , we will use the our training data to 
#create a bayesian classifier.

diabetesCheck = SVC()
diabetesCheck.fit(trainData, trainLabel)

#After we train our bayesian classifier , 
#we test how well it works using our test data.
accuracy = diabetesCheck.score(testData,testLabel)
print("accuracy = ",accuracy * 100,"%")


# In[33]:


diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData,trainLabel)
accuracy = diabetesCheck.score(testData,testLabel)
print("accuracy = ",accuracy * 100,"%")


# In[ ]:





# In[34]:


coeff = list(diabetesCheck.coef_[0])
coeff


# In[35]:


labels = list(dfTrain.drop('Outcome',1).columns)
labels


# In[36]:


features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(11, 6),color = features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')


# In[37]:


#model saving and loading
joblib.dump(diabetesCheck, 'diabeteseModel.pkl')
diabetesLoadedModel = joblib.load('diabeteseModel.pkl')


# In[38]:


#testing loaded model to make prediction
accuracyModel = diabetesLoadedModel.score(testData,testLabel)
print("accuracy = ",accuracyModel * 100,"%")


# In[39]:


dfCheck.head()


# In[40]:


sampleData = dfCheck[:1]
sampleDataFeatures = np.asarray(sampleData.drop('Outcome',1))
sampleDataFeatures


# In[41]:


prediction = diabetesLoadedModel.predict(sampleDataFeatures)
predictionProbab = diabetesLoadedModel.predict_proba(sampleDataFeatures)


# In[42]:


prediction


# In[ ]:




