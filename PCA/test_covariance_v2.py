#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[ ]:





# In[3]:


# Scale function
def scale(vec_1):
    
    # Test
    v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

    v1 = np.multiply(1/np.sqrt(v1_scale + 1e-15),vec_1)
#     print(v1)
    
    return list(v1)
    
    
def QR(span):
    
    # Setup arrays
    # Original Values
    vector = []
    # Normalized Values
    vec_norm = []

    # Size of array
    row,col = span.shape
    
    # First Column
    v1 = span[:,0]
    vec_1 = v1
    vector.append(vec_1)

    v1 = scale(vec_1)
    vec_norm.append(v1)
    
    # Subsequent Columns

    start = 1
    end = col

    for j in range(start,end):

        # Column
        projection = span[:,j]
        v = span[:,j]

        #Orthonormal vector
        for i in range(len(vec_norm)):
            #projection
            # dot
            proj_dot = np.dot(v,vec_norm[i])/(np.dot(vec_norm[i],vec_norm[i]) + 1e-15)
            proj = np.multiply(proj_dot,vec_norm[i])
            projection = projection - proj

        v_norm = scale(projection)

        vec_norm.append(v_norm)
        
    # Calculate R
    Q = vec_norm
    Q = np.array(Q)
    R = np.dot(Q,span)
    R = np.array(R)

    
    return Q.T,R

def eigenspace(span):
    
    eig_vec = np.eye(span.shape[0])
    X = span.copy()

    for _ in range(100):
        Q,R = QR(X)
        eig_vec = np.dot(eig_vec,Q)
        X = np.dot(R,Q)
        eig_val = np.diag(X)
        
    return eig_val,eig_vec, X


# In[4]:


# Dataset for Height
Height = [64,66,68,69,73]
# Dataset for Score
Score = [580, 570, 590, 660, 600]
# Dataset for Age
Age = [29,33,37,46,55]

#Size of array(list)
n = len(Height)


# In[5]:


# Create dataframe.
df = pd.DataFrame([Height, Score, Age])
# df = df.transpose()
# df.columns = ['Height','Score','Age']
print(df)
print(df.shape)
row,col = df.shape

# Dataframe
print(df.cov())


# In[6]:


# # Calculate Mean
# # Height
# height_mean = df['Height'].mean()
# # Score
# score_mean = df['Score'].mean()
# # Age
# age_mean = df['Age'].mean()

# # List of means
# stats = [height_mean,score_mean,age_mean]
# df.iloc[:,0]


# List of means
stats = []
for i in range(col):
    stats.append(df.iloc[:,i].mean())
print(stats)


# In[7]:


Var = []
CoVar = np.zeros((col,col))
print(df.iloc[:,0])

for k in range(df.shape[1]):
    print(stats[k])
    for i in range(df.shape[1]):
        var = 0
        for j in range(df.shape[0]):
            var = var + ((df.iloc[j,k] - stats[k])*(df.iloc[j,i] - stats[i]))/(row-1)
        print(var)
        CoVar[k][i] = var
print(CoVar)

# for i in range(df.shape[1]-1):
#     var = 0
#     for j in range(df.shape[0]):
#         var = var + ((df.iloc[j,i] - stats[0])*(df.iloc[j,i+1] - stats[1]))/(n-1)
#     print(var)
        
# Dataframe
print(df.cov())


# In[8]:


def CoV(df):
    
    # Shape of dataframe
    row,col = df.shape
    print(row,col)
    
    # Determine indices
#     if row > col:
#         row = col
#         col = row
    
    # Initialize Covariance matrix
    CoVar = np.zeros((col,col))
    
    # List of means
    stats = []
    for i in range(col):
        stats.append(df.iloc[:,i].mean())

    # Solve covariance matrix        
    for k in range(col):
        for i in range(col):
            var = 0
            for j in range(row):
                var = var + ((df.iloc[j,k] - stats[k])*(df.iloc[j,i] - stats[i]))/(row-1)
            CoVar[k][i] = var
        
    return CoVar
    
    


# In[9]:


CoVar = CoV(df)
print(CoVar)

