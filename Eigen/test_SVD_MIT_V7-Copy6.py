#!/usr/bin/env python
# coding: utf-8

# In[255]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[256]:


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


# In[257]:


def SVD(A):
    
    # Size of array
    row,col = A.shape
    print('row:', row)
    print('col:', col)
    
    
    # Arrays
    if row < col:
        A1 = A.dot(A.T)
#         print(A1)
    else:
        A1 = A.T.dot(A)
#         print(A1)
        
    # Size of arrays
    row_A,col_A = A1.shape
    print('row_A:', row_A)
    print('col_A:', col_A)
    
    # Eigenspace
    eig_val_A1,eig_vec_A1,x_A1 = eigenspace(A1)
    
    # Eigenvector
    V = eig_vec_A1
#     print('V:',V)
    
    # Singular values
    sigma_A1 = []
    eig_val = eig_val_A1
    for i in range(len(eig_val)):
        sigma_A1.append(np.sqrt(eig_val[i]))
        
    # Sigma
    Sigma = np.sqrt(eig_val_A1) * np.eye(row_A)

    # CV
    if row > col:
        CV = A.dot(V.T)
        CV_0 = CV.copy()
    elif row < col:
        CV = A.T.dot(V.T)
        CV_0 = CV.copy()
        
    
    # Normalize vectors
    # CV = CV.T
#     print(CV)

#     row,col = CV.shape

    for i in range(row):
        CV[i] = CV[i]/(np.linalg.norm(CV[i]) + 1e-15)


    # Solve for U
    U = CV_0.dot(np.linalg.inv(Sigma))
    
    # V
#     if row_A != row and col_A != col:
#         V = V.T

#     if col > row or col < row:
#         V = V.T

#     if col > row:
#         V = V.T
#     elif col < row:
#         V = V.T


        
    return U, Sigma, V
    


# In[258]:


# A = np.array([[1,1,1],[0,1,-1],[-1,1,2],[2,-1,0]])
# A = np.array([[5,5],[-1,7]])
# A = np.array([[1,-8,3,9],[-6,7,-3,5]])
# A = np.array([[4,3,0],[-8,2,-3],[4,5,5],[7,-10,-6]])
# A = np.array([[2,4],[1,3],[0,0],[0,0]])

row = 4
col = 5
A = np.random.randint(-10,10,size=(row,col))

B = A.copy()

print(A)

row,col = A.shape
print(row)
print(col)


# In[259]:


U,E,V = np.linalg.svd(B)
print(U)
print(E)
print(V)


# In[260]:


U,Sigma,V = SVD(B)
print(U)
print(Sigma)
print(V)
# print('\n')
# print(U.T.dot(U))


# In[261]:


print(B)
row,col = B.shape
print(row)
print(col)

# A_0 = np.dot(Sigma.T,V.T)
# A = U.dot(A_0).T
# A = A_0.dot(U)
# print(A)

if col < row:
    
    print('A')

    A_0 = np.dot(Sigma.T,V.T)
    # print(A_0)
    print(U)

#     A = U.dot(A_0).T
    A = U.dot(Sigma).dot(V).T
    print(A)
    
elif col > row:
    
    print('B')
    
    A_0 = np.dot(Sigma.T,V.T)
    # print(A_0)
    print(U)

#     A = U.dot(A_0)
    A = U.dot(Sigma).dot(V)
    print(A)

