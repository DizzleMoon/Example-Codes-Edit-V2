#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
# np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[2]:


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


# In[3]:


# A = np.array([[4,4],[-3,3]])
# A = np.array([[2,4],[1,3],[0,0],[0,0]])

row = 6
col = 5
A = np.random.randint(-10,10,size=(row,col))

row,col = A.shape

if col > row:
    np.set_printoptions(precision=0)
    

B = A.copy()

print(A)


# In[4]:


A1 = A.dot(A.T)
print(A1)
A2 = np.dot(A.T,A)
print(A2)


# In[5]:


# Eigenvalues
eig_val_A1,eig_vec_A1,x_A1 = eigenspace(A1)
print(eig_val_A1)
print(eig_vec_A1)
print(x_A1)

print('\n')

eig_val_A2,eig_vec_A2,x_A2 = eigenspace(A2)
print(eig_val_A2)
print(eig_vec_A2)
print(x_A2)

print('\n')

V = eig_vec_A2.T
print('V:',V)


# In[6]:


#singular values
sigma_A2 = []
eig_val = eig_val_A2
for i in range(len(eig_val)):
    sigma_A2.append(np.sqrt(eig_val[i]))
    
print(sigma_A2)


# In[7]:


Sigma = np.sqrt(eig_val_A2) * np.eye(len(sigma_A2))
print(Sigma)
# V = [v1,v2]
V = eig_vec_A2
V = np.array(V)
print(V)
print(V.T)


# In[8]:


Sigma_V = Sigma.dot(V.T)
print(Sigma_V)

U = A.dot(np.linalg.pinv(Sigma_V))
print(U)

A = U.dot(Sigma).dot(V.T)
print(A)

print(B)


# In[9]:


# # Normalize vectors
# # CV = CV.T
# print(CV)

# row,col = CV.shape

# for i in range(row):
#     CV[i] = CV[i]/(np.linalg.norm(CV[i]) + 1e-15)
    
# print(CV)


# In[10]:


# # Solve for U
# print(Sigma)
# U = CV.dot(np.linalg.pinv(Sigma))
# U1 = U
# print(U1)
# # print(V)
# # print(V.T)


# In[ ]:




