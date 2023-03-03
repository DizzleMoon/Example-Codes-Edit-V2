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
            proj_1 = leng/(proj_0 + 1e-20)
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


# In[3]:


def QR_eig(A):
    row,col = A.shape
    X=A.copy()
    Q,R = QR(X)
#     Q,R = QR(A)
    # print(Q)
    # print(R)

    S = Q
    print(S)
    X = R.dot(Q)

    # print(A)

#     for _ in range(100):
#         Q,R = QR(X)
#         S = S.dot(Q)
#         X = R.dot(Q)

    cnt = 0
            
    while cnt >= 0:
        Q,R = QR(X)
        S = S.dot(Q)
        X = R.dot(Q)
        if np.linalg.norm(np.ones((row,1))) == np.linalg.norm(np.diag(Q)):
            break
        elif cnt > 200:
            break
            
        cnt += 1
                
    return X,S,Q,cnt


# In[4]:


# A = np.array([[2,1,0],[1,3,-1],[0,-1,6]])
A = np.array([[2,4],[1,3],[0,0],[0,0]])
row,col = A.shape
if row != col:
    A = A.dot(A.T)
else:
    A = A
eig_val,eig_vec,Q,cnt = QR_eig(A)
print(eig_val)
print(eig_vec)
print(Q)
print(cnt)
np.linalg.norm(np.ones((3,1)))


# In[5]:


# A = np.array([[2,1,0],[1,3,-1],[0,-1,6]])
# # print(A)

# Q,R = QR(A)
# # print(Q)
# # print(R)

# S = Q
# A = R.dot(Q)

# # print(A)

# for _ in range(100):
#     Q,R = QR(A)
#     S = S.dot(Q)
#     A = R.dot(Q)
    

# print(A)
# print('\n')
# print(Q)
# print('\n')
# print(S)


# In[6]:


# A = np.array([[2,1,0],[1,3,-1],[0,-1,6]])
w,v = np.linalg.eig(A)
print(w)
print(v)


# In[ ]:




