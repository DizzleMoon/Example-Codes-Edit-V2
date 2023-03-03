#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[2]:


# span = np.array([[1,2,3],[-1,1,1],[1,1,1],[1,1,1]])
span = np.array([[2,1,1],[1,3,2],[1,0,0]])
print(span)
A1 = span

row,col = span.shape
print(row,col)


# In[3]:


# Setup arrays
# Original Values
vector = []
# Normalized Values
vec_norm = []


# In[4]:


# Scale function
def scale(vec_1):
    
    # Test
    v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

    v1 = np.multiply(1/np.sqrt(v1_scale),vec_1)
#     print(v1)
    
    return list(v1)
    
    


# In[5]:


def QR(span):
    
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
            proj_dot = np.dot(v,vec_norm[i])/np.dot(vec_norm[i],vec_norm[i])
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


# In[6]:


Q,R = QR(span)
print(Q)
print('\n')
print(R)


# In[7]:


# # span = span.T
# eig_val = np.eye(span.shape[0])
# X = span.copy()

# for i in range(3):
#     print(i)
#     print('\n')
#     Q,R = QR(X)
# #     print('Q:')
# #     print(Q.T)
# #     print('\n')
# #     Q = Q[0:row,0:row]
#     print('Q:')
#     print(Q.T)
#     print('\n')
# #     Q1 = Q[-3:]
# #     print('Q1:')
# #     print(Q1.T)
# #     print('\n')    
#     R = R[0:row,0:row]
#     print('R0:')
#     print(R)
#     print('\n')
#     X = np.dot(R,Q)
#     print('X:')
#     print(X.T)
#     print('/n')


# In[8]:


# span = span.T
eig_val = np.eye(span.shape[0])
X = span.copy()

for i in range(100):
#     print(i)
#     print('\n')
    Q,R = QR(X)
#     print('Q:')
#     print(Q.T)
#     print('\n')
    Q = Q[0:row,0:row]
#     print('Q:')
#     print(Q.T)
#     print('\n')
#     Q1 = Q[-3:]
#     print('Q1:')
#     print(Q1.T)
#     print('\n')    
    R = R[0:row,0:row]
#     print('R0:')
#     print(R)
#     print('\n')
    X = np.dot(R,Q)
    

print(Q)
print('\n')
print(R)
print('\n')    
print('X:')
print(X.T)
print('\n')

