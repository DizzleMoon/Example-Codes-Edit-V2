#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt
import math 
import pandas as pd


# In[2]:


# span = np.array([[1,2,3],[-1,1,1],[1,1,1],[1,1,1]])
# print(span)
# A1 = span


# In[3]:


row = 5
col = 5
span = np.random.randint(10,size=(row,col))
print(span)
A1 = span


# In[4]:


row,col = span.shape
row


# In[5]:


v1 = span[:,0]
v2 = span[:,1]
v3 = span[:,2]
v4 = span[:,3]
v5 = span[:,4]


# In[6]:


# Setup arrays
# Original Values
vector = []
# Normalized Values
vec_norm = []


# In[7]:


# First Column

v1 = span[:,0]
vec_1 = v1
print(vec_1)
vector.append(vec_1)

# Test
v1_scale = np.square(np.linalg.norm(vec_1))
print(v1_scale)

v1 = np.multiply(1/np.sqrt(v1_scale),vec_1)
print(v1)
vec_norm.append(v1)


# In[8]:


def QR(span): 
    
    # Copy matrix
    A1 = span
    
    # Size of array
    row,col = span.shape
    
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

        v_norm = np.multiply(1/np.sqrt(v_scale + 1e-20),vect_0)
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
#     print(R)
    
    return Q.T, R


# In[9]:


# Function
Q,R = QR(A1)


# In[10]:


print(Q)
print("\n")
print(R)


# In[11]:


# # Calculate R
# Q = vec_norm
# # A1 = np.multiply(1,A1)
# R = np.dot(Q,A1)
# print(R)


# In[12]:


# Function
C = np.array([[2,1,0],[1,3,1],[0,1,4]])
Q,R = QR(C)
print(Q)
print(R)

A2 = np.dot(Q,R)
print(A2)


# In[13]:


A = np.array([[2,1,0],[1,3,1],[0,1,4]])
pQ = np.eye(3)
X=A.copy()
Y=A.copy()
for i in range(100):
        Q,R = QR(X)
        pQ = pQ.dot(Q)
        X = R.dot(Q)
        
print(pQ)
print(X)


# In[14]:


np.linalg.eig(Y)

