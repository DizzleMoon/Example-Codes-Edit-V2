#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[10]:


# X =  np.array([1,0,0,1,1,0,1,1,1]).reshape(3,3)
X =  np.array([1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1]).reshape(4,4)
# row = 4
# col = 4
# X = np.random.randint(2,size=(row,col))
Y = X.copy()
X
row,col = X.shape
print(X)
Y = X.copy()
print(X)


# In[11]:


Q,R = np.linalg.qr(X)
Q1 = Q.copy()

print(Q)
print('\n')
print(R)

print('\n')

X =Q.dot(R)
print(X)

print('\n')

# np.dot(Q[row-1],Q[row-1].T)


# In[12]:


# Initialize arrays

U = []
V_norm = []
U_Norm = []
A = []
projection = []


# In[13]:


# First Column

a1 = X[:,0]
A.append(a1)
u1 = a1
print(u1)
U.append(u1)

# Normalization
a1_norm = np.linalg.norm(a1)
u1_normalized = a1/a1_norm
print(u1_normalized)
V_norm.append(u1_normalized)


# In[14]:


# Subsequent Columns
for j in range(1,row):
    a = X[:,j]
    u = a
    A.append(a)
    
    # Projection
    projection = []
    for i in range(j):

        # Dot products
        # Denominator
        denom = U[i-1].dot(U[i-1].T)
        # Numerator
        num = U[i-1].dot(a)
        # Multiple
        vec = np.multiply(num/denom,U[i-1])
        projection.append(vec)
        
#     print(projection)
    
    for k in range(len(projection)):
        u1 = a - projection[k]
        a = u1
#     print(a)
    
    U.append(a)
    
    # Normalization
    v_norm = np.linalg.norm(a)
    v_normalized = a/v_norm
    print(v_normalized)
    V_norm.append(v_normalized)

print('\n')
# U_norm = []
# for i in V_norm:
#     U_norm.append(list(i))
# print(U_norm)
# print(V_norm[0])
# print('\n')
# print(U)


# In[15]:


err = []
for i in range(len(U)):
    err.append(np.dot(U[i],U[i]))
    
print(err)

# print(V_norm[3].T.dot(V_norm[3]))


# In[16]:


for i in range(row):
    Q1[i,:] = np.multiply(-1,Q1[i,:])
print(Q1)
print('\n')
print(err)
print('\n')
print(V_norm)


# In[ ]:




