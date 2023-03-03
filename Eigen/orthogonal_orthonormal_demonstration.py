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


vec = np.array([[1,0,-1],[1,np.sqrt(2),1],[1,-np.sqrt(2),1]])
print(vec.T)

# Normalize
vec[0] = vec[0]/np.linalg.norm(vec[0])
vec[1] = vec[1]/np.linalg.norm(vec[1])
vec[2] = vec[2]/np.linalg.norm(vec[2])

print(vec)
print(np.linalg.norm(vec[0]))

vec.T.dot(vec)


# In[3]:


dot1 = vec[1].dot(vec[2])
print(dot1)


# In[4]:


u1 = vec[0]/np.linalg.norm(vec[0])
print(u1)
u2 = vec[1]/np.linalg.norm(vec[1])
print(u2)
u3 = vec[2]/np.linalg.norm(vec[2])
print(u3)
print(np.linalg.norm(vec[0]))


# In[5]:


# Orthogonal & Orthonormal Check
# Orthogonal
orthogonal = u1.dot(u3)
print(orthogonal)
# Orthonormal
orthonormal = np.linalg.norm(u1)
print(orthonormal)


# In[6]:


u = np.array([1,2,0])
v = np.array([0,0,3])
y = np.array([2,-2,3,-4])

# Check for orthogornality
uv = u.dot(v)
print(uv)

# Check for orthonormality
u_norm = np.linalg.norm(u)
print(u_norm)
v_norm = np.linalg.norm(v)
print(v_norm)
y_norm = np.linalg.norm(y)
print(y_norm)

# Normalization
u1_norm = u/np.linalg.norm(u)
print(u1_norm)
u2_norm = v/np.linalg.norm(v)
print(u2_norm)

# Magnitudes(Normalization) of normalized vecotrs
u1_norm_mag = np.linalg.norm(u1_norm)
print(u1_norm_mag)
u2_norm_mag = np.linalg.norm(u2_norm)
print(u2_norm_mag)

# Cosine similarity
cos_uv = (uv)/(u_norm*v_norm)
print(cos_uv)


# In[7]:


Q = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
print(Q)

# Check for orthogonality
Q_ortho = Q.T.dot(Q)
print(Q_ortho)


# In[ ]:




