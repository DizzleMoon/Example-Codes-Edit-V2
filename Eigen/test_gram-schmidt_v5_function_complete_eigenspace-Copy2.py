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


# span = np.array([[1,2,3],[-1,1,1],[1,1,1],[1,1,1]])
# print(span)
# A1 = span


# In[3]:


row = 3
col = 3
span = np.random.randint(10,size=(row,col))
print(span)
A1 = span
A2 = A1


# In[4]:


row,col = span.shape
row


# In[5]:


# v1 = span[:,0]
# v2 = span[:,1]
# v3 = span[:,2]
# v4 = span[:,3]
# v5 = span[:,4]


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
#     print(R)
    
    return Q.T,R


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



# for _ in range(20):
#     Q,R = QR(A2)
#     Q = np.array(Q)
#     R = np.array(R)
#     A2 = np.dot(Q.T,np.dot(A2,Q))
    
# print(A2)


# In[13]:


pQ = np.eye(row)
X = A2
for i in range(100):
        Q,R = QR(X)
        pQ = pQ @ Q
        X = R @ Q
print(np.diag(X))
print(pQ)


# In[14]:


w,v = LA.eig(A2)
print(w)
print(v)


# In[15]:


B = np.array([[1,2,3],[3,2,1],[1,0,-1]])
B_diag = np.diag(B)


# In[16]:


# for _ in range(20):
#     Q,R = QR(B)
#     Q = np.array(Q)
#     R = np.array(R)
#     B = np.dot(R,Q)
# #     B = np.dot(Q.T,np.dot(B,Q))
# B


# In[17]:


w,v = np.linalg.eig(B)
w


# In[18]:


# a = np.array([[0, 2], 
#               [2, 3]])
# p = [1, 5, 10, 20]
print(B)
for i in range(20):
    q, r = QR(B)
    q = np.array(q)
    B = np.dot(r,q)
#     B = np.dot(q.T,np.dot(B,q))
#     if i+1 in p:
#         print(f'Iteration {i+1}:')
#         print(A2)
        
A2_diag = np.diag(B)
print(A2_diag)


# In[19]:


# C = np.array([[2,1,0],[1,3,1],[0,1,4]])
# C1 = C
# print(C)
# # q,r = QR(C)
# pQ = np.eye(C.shape[1])
# for _ in range(100):
#     q,r = QR(C)
#     pQ = pQ@q
#     C = np.dot(r,q)

# # print(q)

# C_diag = np.diag(C)
# print(C_diag)
# print('\n')
# print(pQ)


# In[20]:


# pQ = np.eye(C.shape[0])
# for i in range(100):
#     Q,R = QR(C)
#     C = R@Q
#     pQ = pQ@Q
    
# print(np.diag(C))
# print(pQ)


# In[21]:


# C = np.array([[2,1,1],[1,3,2],[1,0,0]])
C = np.array([[2,1,0],[1,3,1],[0,1,4]])
C11 = C

# Q,R = np.linalg.qr(C)
# print(Q)

# Q,R = QR(C)
# print(Q)

# p_eye = np.eye(C.shape[0])
# for _ in range(100):
#     Q,R = QR(C)
#     p_eye = np.dot(p_eye,Q)
#     C = np.dot(R,Q)

    
# print(np.diag(C))
# print(p_eye)

pQ = np.eye(C.shape[0])
X=C.copy()
X1 = C.copy()
for i in range(100):
        Q,R = QR(X)
        Q = np.array(Q)
        pQ = np.dot(pQ,Q)
        X = np.matmul(R,Q)
        
print(np.diag(X))
print(pQ)


# In[22]:


w,v = np.linalg.eig(X1)
print(w)
print(v)


# In[23]:


# p_eye = np.eye(C1.shape[0])
# print(p_eye)
# print(w)
# i_eye = np.multiply(w,p_eye)
# i_eye


# In[24]:


# M = C1 - i_eye
# M


# In[25]:


# pQ = np.eye(C1.shape[0])
# X=X1=C1.copy()
# for i in range(100):
#         Q,R = np.linalg.qr(X)
#         pQ = pQ @ Q;
#         X = R @ Q;


# In[26]:


# print(pQ)
# print(np.diag(X))


# In[27]:


# w,v = np.linalg.eig(X1)
# print(w)
# print(v)


# In[28]:


D = np.array([[5.4,4,7.7],[3.5,-0.7,2.8],[-3.2,5.1,0.8]])
D_q,D_r = QR(D)
D_q
D_r


# In[29]:


pQ = np.eye(D.shape[0])
X=D1=D.copy()
for i in range(100):
        Q,R = np.linalg.qr(X)
        pQ = pQ @ Q
        X = R @ Q
print(X)
print(pQ)


# In[30]:


for i in range(100):
        Q,R = QR(X)
        pQ = pQ @ Q.T
        X = R @ Q
print(np.diag(X))
print(pQ)


# In[31]:


pQ = np.eye(3)
for i in range(100):
        Q,R = QR(X)
        pQ = pQ.dot(Q.T)
        X = R.dot(Q)
print(np.diag(X))
print(pQ)


# In[32]:


np.linalg.eig(D1)


# In[ ]:




