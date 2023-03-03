#!/usr/bin/env python
# coding: utf-8

# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import math 


# In[12]:


A2 = np.array([[1,1,0],[0,1,1],[0,2,1],[0,0,3]])
print(A2)


# In[13]:


# Split columns
u1 = A2[:,0]
a1 = u1
u2 = A2[:,1]
a2 = u2
u3 = A2[:,2]
a3 = u3


# In[14]:


# Partial Basis
X_partial = a1
# First Column
v1 = a1
print(v1)


# In[15]:


# Second Column
# V2 / Second Column
proj_dot_2 = np.dot(a1,a2)
proj_2 = proj_dot_2/np.sum(a1 ** 2)
print(proj_2)
# a2_1 = np.multiply(27,a2)
# v2 = np.dot(1/proj_2,a1)
v2 = np.dot(1/proj_2,a2) - a1
print(v2)

# v2_sq = np.square(np.linalg.norm(v2))
# print(v2_sq)


# v2 = a2 - np.dot(proj_2,a1)
# print(v2)

# Second column
# v2_scale = np.multiply(1/np.square(np.linalg.norm(v2)),v2)
# v2_scale

# Orthogornal Test
np.dot(v1,v2)


# In[16]:


# Third Column
# V3 / Third Column
v3 = a3
print(v3)

proj_dot_2 = np.dot(a1,a3)
print(proj_dot_2)
length_1 =  np.sum(a1 ** 2)
print(length_1)
proj_2 = proj_dot_2/length_1
# proj_2 = 1/(proj_2 + 1e-5)
# proj_2 = proj_dot_2/np.sum(a1 ** 2)
print(proj_2)


proj_dot_3 = np.dot(v2,a3)
print(proj_dot_3)
length_2 = np.sum(v2 ** 2)
print(length_2)
proj_3 = proj_dot_3/length_2
print(proj_3)

# print(proj_3/proj_2)

# v3 = a3 - np.dot(proj_2,a1) - np.dot(proj_3,v2)
# v3 = np.dot(proj_3,a3) - np.dot((proj_3/proj_2),a1) - np.dot(proj_3,v2)
v3_0 = np.multiply(length_2,a3)
print(v3_0)
# proj_a = proj_3/proj_2
# print(proj_a)
v3_1 = np.multiply(proj_2,a1)
print(v3_1)
print(v2)
print('proj_3:',proj_3)
v3_2 = np.multiply(proj_3*length_2,v2)
print(v3_2)
# print('v3: \n',v3)
v3 = v3_0 - v3_1 - v3_2
print(v3)

# # v3_scale = np.multiply(10,v3)
# # # v3_scale = np.dot(10,a3) - np.dot(10, np.dot(proj_2,a3)) - np.dot(10, np.dot(proj_3,a3))
# # print(v3_scale)
# # # print(a3)

# Check for orthogonality
print("\n")
print(np.dot(v2,v3))
print(np.dot(v1,v3))
print(np.dot(v1,v2))


# In[17]:


# Q = [v1,v2,v3]
# print(Q)


# In[18]:


# For an orthonormal basis: normalize at the end
# First column
v1_scale = np.square(np.linalg.norm(v1))
print(v1_scale)
# Second column
v2_scale = np.square(np.linalg.norm(v2))
print(v2_scale)
# Thrid column
v3_scale = np.square(np.linalg.norm(v3))
print(v3_scale)


# In[19]:


# Assemble Q
v1 = np.multiply(1/np.sqrt(v1_scale),v1)
v2 = np.multiply(1/np.sqrt(v2_scale),v2)
v3 = np.multiply(1/np.sqrt(v3_scale),v3)

Q = np.array([v1,v2,v3])
# print(Q)
print(Q.T)


# In[20]:


# Calculate R
R = np.dot(Q,A2)
print(R)


# In[ ]:




