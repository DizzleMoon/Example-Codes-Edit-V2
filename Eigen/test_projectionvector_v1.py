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


a = np.array([2,1,4])
b = np.array([1,-2,1])

# Cpmpare vectors for projecion
compare_a_b = a.dot(b)/np.linalg.norm(a)
print(compare_a_b)

# Projection a onto b v1
proj_a_b = ((a.dot(b))/(np.linalg.norm(b))**2)*b
print(proj_a_b)
print(np.linalg.norm(proj_a_b))

# Projection a onto b v2
proj_a_b = a.dot(b)/b.T.dot(b)
print(proj_a_b)


# In[4]:


# Math Syd
u = np.array([4,0,-3])
v = np.array([1,-2,2])


# dot product
uv = u.dot(v)
print(uv)

# Normalize vectors
u_norm = np.linalg.norm(u)
print(u_norm)
v_norm = np.linalg.norm(v)
print(v_norm)

# The component of v in the direction of u is
comp_v_u = uv/u_norm
print(comp_v_u)
# The projection of v in the direction of u is
proj_v_u = uv/u_norm**2*u
print(proj_v_u)

# Orthonormal Projection
ortho = v - proj_v_u
print(ortho)


# In[ ]:




