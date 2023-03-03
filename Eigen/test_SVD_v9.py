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


def SVD(A):
    
    # Copy of array
    B = A.copy()
    
    # Size of array
    row,col = A.shape
    
    A1 = A.dot(A.T)
    print(A1)
    A2 = np.dot(A.T,A)
    print(A2)
    
    if row < col:
        # Eigenvalues
        eig_val_A1,eig_vec_A1,x_A1 = eigenspace(A1)
        
        # U: left Singular Array
        U = eig_vec_A1

        # Sigma: Middle Singular Array
        # Initialize Sigma array
        Sigma = np.zeros((row,col))
        # Initialize Eigenvalues array
        Eig = eig_val_A1*np.eye(len(eig_val_A1))

        # Create Sigma
        if row < col:
            sig = row
        elif row > col:
            sig = col
        Sigma[0:sig,0:sig] = Eig[0:sig,0:sig]
        Sigma = np.sqrt(Sigma)

        # Solve for V: Right Singular Array
        # Initialize V
        V = np.zeros((col,col))

        for i in range(len(U)):

            u1 = A.T.dot(U[:,i])
            # Normalize
            u1 = u1/(np.linalg.norm(u1) + 1e-20)

            # Update V array
            V[i,:] = u1

        # Determine V
        V = A.T.dot(U).dot(Sigma)

        for i in range(row):
            V[:,i] = V[:,i]/(np.linalg.norm(V[:,i]) + 1e-20)
            
    else:
        # Eigenvalues
        eig_val_A1,eig_vec_A1,x_A1 = eigenspace(A1)
        eig_val_A2,eig_vec_A2,x_A2 = eigenspace(A2)
        
        # Right Singular Vector
        V = eig_vec_A2
        
        # Sigma
        # Initialize Sigma array
        Sigma = np.zeros((row,col))
#         print(Sigma)
        # Initialize Eigenvalues array
        Eig = eig_val_A2*np.eye(len(eig_val_A2))
#         print(Eig)

        # Create Sigma
        if row < col:
            sig = row
        else:
            sig = col
        Sigma[0:sig,0:sig] = Eig[0:sig,0:sig]
        Sigma = np.sqrt(Sigma)
#         print(Sigma)
        
        # Solve for U
        # Initialize U
        # U = np.eye((row))
        U = np.zeros((row,row))
#         print(U)

        eig = eig_val_A2
        for i in range(len(eig)):    

            u1 = (1/(np.sqrt(eig[i]) + 1e-20))*A.dot(V[:,i])
#             print(u1)

            u1 = u1/(np.linalg.norm(u1) + 1e-20)
#             print(u1)

            U[:,i] = u1
        
        
        
    return U,Sigma,V
    


# In[4]:


# A = np.array([[1,1],[1,1],[1,-1]])
# A = np.array([[2,4],[1,3],[0,0],[0,0]])
# A = np.array([[-5,1,-2,5],[0,-2,4,8]])

row = 5
col = 6
A = np.random.randint(-10,10,size=(row,col))

row,col = A.shape

print(A)

B = A.copy()


# In[5]:


U,Sigma,V = SVD(B)
print(U)
print(Sigma)
print(V)


# In[6]:


print(B)
A = U.dot(Sigma).dot(V.T)
print(A)

# Check for identity matrices
U_test = U.T.dot(U)
print(U_test)


# In[ ]:




