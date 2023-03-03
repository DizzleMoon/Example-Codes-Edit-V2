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
            proj_1 = leng/(proj_0 + 1e-15)
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


def QR_Givens(A):
    # Initialize G list
    G_lst = []
    Q = np.eye(row)
    
    for j in range(0,row-1):
        for i in range(j+1,row):
            # Initialize Given Rotation Matrix
            G = np.eye(row)

            # Initialize variables
            x = A[j,j]
            y = A[i,j]
            r = np.sqrt(x**2 + y**2)

            r1 = np.linalg.norm(A[j:i,j])

            cos_t = x/r
            sin_t = y/r

            G[j,j] = cos_t
            G[j,i] = sin_t
            G[i,j] = -sin_t
            G[i,i] = cos_t

            G_lst.append(G)

            Q = np.dot(Q,G)

            A = np.dot(G,A)
            
    #Q
    G_lst_len = len(G_lst)
    G_lst = np.array(G_lst)
    # Q_mat = np.dot(G1.T,np.dot(G2.T,G3.T))
    Q_mat = G_lst[0].T
    for i in range(1,G_lst_len):
    #     print(Q_mat)
        Q_mat = Q_mat.dot(G_lst[i].T)
#     print(Q_mat)

    return Q_mat,A

 


# In[4]:


def QR_eig(A):
    pQ = np.eye(A.shape[0])
#     print('pQ:',pQ)
    X=A.copy()
    for i in range(100):
            Q,R = QR_Givens(X)
            pQ = pQ.dot(Q)
            X = R.dot(Q)
            
                
    return pQ,X


# In[5]:


row = 4
col = 2
A = np.random.randint(10,size=(row,col))
print('A:',A)

# A = np.array([[2,4],[1,3],[0,0],[0,0]])
# A = np.array([[2,1,0],[1,3,1],[0,1,4]])
# W = A.dot(A.T)
row,col = A.shape
if row == col:
    W = A.copy()
else:
    W = A.dot(A.T)
print(W)


# In[6]:


# W = W.dot(W.T)
eigen_vec,eigen_val = QR_eig(W)
print(eigen_val)
print('\n')
print(eigen_vec)

r,c = A.shape
print(r,c)

idty = np.eye(r)
print(idty)

# for i in range(c):
#     for j in range(r):
#         idty[j,i] = eigen_vec[j,i]

print(idty)


# In[7]:


w,v = np.linalg.eig(W)
print(w)
print(v)


# In[8]:


import scipy.linalg as la
eigvals, eigvecs = la.eig(W)
print(eigvals.real)
print(eigvecs)


# In[9]:


# a = np.array([[0, 2], 
#               [2, 3]])
a = b = W.copy()
p = [1, 5, 10, 20]
for i in range(20):
    q, r = np.linalg.qr(a)
    a = np.dot(r, q)
    if i+1 in p:
        print(f'Iteration {i+1}:')
        print(a)


# In[10]:


def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n


# In[11]:


x = np.diag(eigen_val)
a = b

for i in range(100):
    x = np.dot(a, x)
    lambda_1, x = normalize(x)
    
print(lambda_1)
print(x)


# In[12]:


# power iteration / von Mises iteration
# gives an approximation for an eigenvector for the dominant eigenvalue of a matrix A
# by doing the power iteration k times
def power_iteration(A, k: int):
    v = np.random.rand(A.shape[1])

    for _ in range(k):
        # calculates b = Av
        b = np.dot(A, v)

        # calculate the norm of b
        b_norm = np.linalg.norm(b)

        # define v to be the normalized version of b
        v = b / b_norm

    return v

def give_eigenvalue(A,k):
    v=power_iteration(A,k)
    lam = np.dot(A,v)[0]/v[0]
    
    return lam


# In[13]:


v=power_iteration(b,3)
v

