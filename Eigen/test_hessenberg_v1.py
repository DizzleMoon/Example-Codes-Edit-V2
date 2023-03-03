#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA


# In[2]:


A = np.array([[1,2,3,4],[2,1,2,2],[3,2,1,3],[4,2,3,1]])
print(A)


# In[3]:


# Shape
n,m = A.shape


# In[4]:


for i in range(0,n-1):
    x = A[i+1:n,i]
    x1 = np.dot(-np.sign(A[i+1,i]),np.sqrt(np.dot(x.T,x)))
    u = np.zeros((n-i-1,1))
    u[0] = np.sqrt((x1-x[0])/(2*x1))
    for j in range(1,n-i-1):
        u[j] = x[j]/(-2*u[0]*x1)

    H = np.eye(n)
    H[i+1:n,i+1:n] = H[i+1:n,i+1:n] - np.dot(2,np.dot(u,u.T))
    print(H)
    A = np.dot(H.T,np.dot(A,H))
    print(A)
A


# In[5]:


def hess(A):
    
    # Shape
    n,m = A.shape
    
    for i in range(0,n-1):
        x = A[i+1:n,i]
        x1 = np.dot(-np.sign(A[i+1,i]),np.sqrt(np.dot(x.T,x)))
        u = np.zeros((n-i-1,1))
        u[0] = np.sqrt((x1-x[0])/(2*x1))
        for j in range(1,n-i-1):
            u[j] = x[j]/(-2*u[0]*x1)

        H = np.eye(n)
        H[i+1:n,i+1:n] = H[i+1:n,i+1:n] - np.dot(2,np.dot(u,u.T))
        print(H)
        A = np.dot(H.T,np.dot(A,H))
        print(A)

    return A


# In[6]:


A = np.array([[1,2,3,4],[2,1,2,2],[3,2,1,3],[4,2,3,1]])
A = hess(A)
print(A)

