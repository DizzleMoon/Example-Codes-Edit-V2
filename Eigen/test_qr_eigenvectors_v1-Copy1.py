#!/usr/bin/env python
# coding: utf-8

# In[15]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[16]:


vector = []


# In[17]:


X = np.array([[1,2,3],[-1,1,1],[0,0,0],[0,0,0]])
# X = X.T
print(X)
print(X.dot(X.T))


# In[18]:


v1 = X[:,0]
print(v1)
vector.append(v1)


# In[19]:


a = np.dot(X[:,1],X[:,0])
b = np.dot(X[:,0],X[:,0])
print(a)
print(b)
proj = (a/b) * b
v2 = np.multiply(b,X[:,1]) -np.multiply(proj,X[:,0])
print(v2)
vector.append(v2)

# Check
check = v1.dot(v2)
print(check)


# In[20]:


# denom = []
# a = np.dot(X[:,0],X[:,0])
# denom.append(a)
# b = np.dot(X[:,2],X[:,0])
# proj_1 = b/a
# c = np.dot(v2,X[:,2])
# d = np.dot(v2,v2)
# denom.append(d)
# proj_2 = c/d
# # c = np.dot(X[:,2],X[:,0])

# print(denom)
# print(a)
# print(b)
# print(c)
# print(d)
# # num = max(denom)
# print(num)
# print(np.multiply(proj_1*num,X[:,0]))

# # Multiple
# num = max(denom)/min(denom)

# # v2 = np.multiply(b,X[:,1]) -np.multiply(proj,X[:,0])
# v3 = np.multiply(num,X[:,2]) - np.multiply(proj_1*num,X[:,0]) - np.multiply(proj_2*num,v2)
# print(v3)
# vector.append(v3)
# print(vector)
# vector = np.array(vector)
# print(vector.T)


# In[21]:


# denom = []
# a = np.dot(X[:,0],X[:,0])
# denom.append(a)
# b = np.dot(X[:,2],X[:,0])
# proj_1 = b/a
# c = np.dot(v2,X[:,2])
# d = np.dot(v2,v2)
# denom.append(d)
# proj_2 = c/d
# # c = np.dot(X[:,2],X[:,0])

# print(denom)
# print(a)
# print(b)
# print(c)
# print(d)
# # num = max(denom)
# print(num)
# print(np.multiply(proj_1*num,X[:,0]))

# # Multiple
# # num = max(denom)/min(denom)
# num = max(denom)

# # v2 = np.multiply(b,X[:,1]) -np.multiply(proj,X[:,0])
# v3 = np.multiply(num,X[:,2]) - np.multiply(proj_1*num,X[:,0]) - np.multiply(proj_2*num,v2)
# print(v3)
# vector.append(v3)
# print(vector)
# vector = np.array(vector)
# print(vector.T)


# In[22]:


# Copy matrix
A1= A2 = X.copy()

# Size of array
row,col = X.shape

# Setup arrays
# Original Values
vector = []
# Normalized Values
vec_norm = []

# First Column
v1 = X[:,0]
vec_1 = v1
#     print(vec_1)
vector.append(vec_1)

# Test
v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

v1 = np.multiply(1/np.sqrt(v1_scale),vec_1)
#     print(v1)
vec_norm.append(v1)     

# Subsequent Columns

start = 1
end = col

for i in range(start,end):

    # Initialize vectors
    length = []
    projection = []
    project = []

    # Columns
    v = X[:,i]

    for i in vector:
        leng = np.dot(i,v)
        length.append(leng)
        proj_0 = np.dot(i,i)
        projection.append(proj_0)
        proj_1 = leng/proj_0
        project.append(proj_1)

#         print(length)
#         print(projection)
#         print(project)

    # Factorization
    a = min(projection)
#         print('a:',a)

    # QR equation
    vec_0 = 0
    vec_1 = np.multiply(a,v)
    for i in range(len(vector)):
        vec_0 = vec_0 - np.multiply(project[i]*a,vector[i])

    vect_0 = vec_1 + vec_0    
#         print(vect_0)
    vector.append(vect_0)

    # Scaling vector    
    # Scale
    v_scale = np.square(np.linalg.norm(vect_0))
#         print(v_scale)

    v_norm = np.multiply(1/np.sqrt(v_scale + 1e-15),vect_0)
#         print(v_norm)

    v = v_norm
    vec_norm.append(v_norm)
#         print('v norm:', v_norm)
#         print('\n')
#         print(vector)
#         print('\n')
#         print(vec_norm)
#         print('\n')
#         print(np.array(vec_norm).T)

# Calculate R
# Q matrix
Q = np.array(vec_norm)
# A1 = np.multiply(1,A1)
R = np.dot(Q,A1)

print(Q.T)
print(R)

print('\n')

Q1,R1 = np.linalg.qr(A2)
print(Q1)
print(R1)


# In[23]:


def QR(span): 
    
    # Copy matrix
    A1 = span.copy()
    
    # Size of array
    row,col = A1.shape
    
    # Setup arrays
    # Original Values
    vector = []
    # Normalized Values
    vec_norm = []
    
    # First Column
    v1 = span[:,0]
    vec_1 = v1
#     print(vec_1)
    vector.append(vec_1)

    # Test
    v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

    v1 = np.multiply(1/np.sqrt(v1_scale),vec_1)
#     print(v1)
    vec_norm.append(v1)     

    # Subsequent Columns

    start = 1
    end = col

    for i in range(start,end):
        
        # Initialize vectors
        length = []
        projection = []
        project = []

        # Columns
        v = span[:,i]

        for i in vector:
            leng = np.dot(i,v)
            length.append(leng)
            proj_0 = np.dot(i,i)
            projection.append(proj_0)
            proj_1 = leng/(proj_0 + 1e-15)
            project.append(proj_1)

#         print(length)
#         print(projection)
#         print(project)
        
        # Factorization
        a = min(projection)
#         print('a:',a)

        # QR equation
        vec_0 = 0
        vec_1 = np.multiply(a,v)
        for i in range(len(vector)):
            vec_0 = vec_0 - np.multiply(project[i]*a,vector[i])

        vect_0 = vec_1 + vec_0    
#         print(vect_0)
        vector.append(vect_0)

        # Scaling vector    
        # Scale
        v_scale = np.square(np.linalg.norm(vect_0))
#         print(v_scale)

        v_norm = np.multiply(1/np.sqrt(v_scale + 1e-15),vect_0)
#         print(v_norm)

        v = v_norm
        vec_norm.append(v_norm)
#         print('v norm:', v_norm)
#         print('\n')
#         print(vector)
#         print('\n')
#         print(vec_norm)
#         print('\n')
#         print(np.array(vec_norm).T)
        
    # Calculate R
    # Q matrix
    Q = vec_norm
    Q = np.array(Q)
    # A1 = np.multiply(1,A1)
    R = np.dot(Q,A1)
#     print(R)
    
    return Q.T,R


# In[24]:


X = np.array([[1,2,3],[-1,1,1],[0,0,0],[0,0,0],[0,0,0]])
X1 = X.copy()
# X = X.T
X = X.dot(X.T)
Y = X.copy()
Q,R = QR(X1)
print(Q)
print(R)


# In[25]:


w,v = np.linalg.qr(X1)
print(w)
print(v)


# In[26]:


span = Y.copy()

# Copy matrix
A1 = span.copy()

# Size of array
row,col = A1.shape
print('row_A1', row)

# Setup arrays
# Original Values
vector = []
# Normalized Values
vec_norm = []

# First Column
v1 = span[:,0]
vec_1 = v1
#     print(vec_1)
vector.append(vec_1)

# Test
v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

v1 = np.multiply(1/np.sqrt(v1_scale),vec_1)
#     print(v1)
vec_norm.append(v1)     

# Subsequent Columns

start = 1
end = col

for i in range(start,end):
    
#     print('I:',i)

    # Initialize vectors
    length = []
    projection = []
    project = []

    # Columns
    v = span[:,i]

    for i in vector:
        leng = np.dot(i,v)
        length.append(leng)
        proj_0 = np.dot(i,i)
        projection.append(proj_0)
        proj_1 = leng/(proj_0 + 1e-15)
        project.append(proj_1)

    #         print(length)
    #         print(projection)
    #         print(project)

    # Factorization
    a = max(projection)
    #         print('a:',a)

    # QR equation
    vec_0 = 0
    vec_1 = np.dot(a,v)
    for i in range(len(vector)):
        vec_0 = vec_0 - np.multiply(project[i]*a,vector[i])
        
#     print('vec_0',vec_0)
#     if i == 2:
#         vv = [0,0,1,0]
#         vec_0 = vec_0.dot(vv)
#         print('vec_0_3:',vec_0)
#     print('vec_0_2',vec_0)
    vect_0 = vec_1 + vec_0    
#     print('vect_0',vect_0)
    vector.append(vect_0)

    # Scaling vector    
    # Scale
    v_scale = np.square(np.linalg.norm(vect_0))
    #         print(v_scale)

    v_norm = np.multiply(1/np.sqrt(v_scale + 1e-15),vect_0)
    #         print(v_norm)

    v = v_norm
#     print('v:', v)
    vec_norm.append(v_norm)
    #         print('v norm:', v_norm)
    #         print('\n')
    #         print(vector)
    #         print('\n')
    #         print(vec_norm)
    #         print('\n')
    #         print(np.array(vec_norm).T)

# Calculate R
# Q matrix
Q = vec_norm
Q = np.array(Q)
print(Q)
# A1 = np.multiply(1,A1)
R = np.dot(Q,A1)
print(R)

# Q1 = Q[0:2,0:4]
# print(Q1)

Q2 = np.array([[1,0,1],[0,1,1],[0,0,0],[0,0,0],[0,0,0]])
print(Q2)

Q3 = np.dot(Q,Q2)
print(Q3)
# return Q.T,R


# In[27]:


Q = Q.T
row_Q,col_Q = Q3.shape
Q1 = np.multiply(Q,np.eye(row))
print(np.eye(row))

Q_eye = np.eye(row)
Q_eye[0:row_Q,0:col_Q] = Q3
Q_eye


# In[28]:


# sparse = np.array([[1,1,0,0],[1,1,0,0],[1,1,0,1],[1,1,1,0]])
# np.multiply(Q,sparse)


# In[ ]:




