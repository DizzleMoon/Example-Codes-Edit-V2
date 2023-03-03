#!/usr/bin/env python
# coding: utf-8

# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[13]:


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


# In[14]:


C = np.array([[5,5],[-1,7]])
# C = np.array([[2,2],[1,1]])
D = C.copy()

# Solve for svd using numpy
np.linalg.svd(D)


# In[15]:


A2 = np.dot(C.T,C)
print(A2)


# In[16]:


# Eigenspace
eig_val_A2,eig_vec_A2,x_A2 = eigenspace(A2)
Sigma = x_A2*np.eye(2)
print(eig_val_A2)
print(eig_vec_A2)
print(Sigma)


# In[17]:


#singular values
sigma_A2 = []
eig_val = eig_val_A2
for i in range(len(eig_val)):
    sigma_A2.append(np.sqrt(eig_val[i]))
    
print(sigma_A2)


# In[18]:


# Charateristic Polynomial

# Eigenvalues
print(eig_val_A2)

poly = []
for i in range(len(sigma_A2)):
    poly.append(A2 - np.multiply(sigma_A2[i]**2,np.eye(2)))
    
print(poly)
print('\n')

# Eigenvalue #1
eig_1 = poly[0]
print(eig_1)
print('\n')
# Eigenvalue #2
eig_2 = poly[1]
print(eig_2)


# In[19]:


# Eigenvectors
# lamda 2
eig_vec_2 = A2 - eig_val_A2[0]*np.eye(2)
print(eig_vec_2)
# Normalize
v1_0 = eig_1[0]/np.linalg.norm(eig_1[0])
print(v1_0)

# lamda 1
eig_vec_1 = A2 - eig_val_A2[1]*np.eye(2)
print(eig_vec_1)
# Normalize
v2_0 = eig_2[0]/np.linalg.norm(eig_2[0])
print(v2_0)

V = [v2_0,v1_0]
V = np.array(V).T
print(V)


# In[20]:


CV = C.dot(V)
print(CV)


# In[21]:


# Solve for U
print(eig_val_A2)
print(Sigma)
Sig = np.multiply(np.sqrt(eig_val_A2),np.eye(2))
print(Sig)
U = CV.dot(np.linalg.pinv(Sig))
print(U)
print(V)
# print(V.T)

print('\n')

# Check for decomposition
A = U.dot(Sig).dot(V.T)
print(A)
print(D)


# In[22]:


# Solve for svd using numpy
np.linalg.svd(D)


# In[ ]:




