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


# A = np.array([[1,2,3,4],[2,1,2,2],[3,2,1,3],[4,2,3,1]])
# A = np.array([[6,5,7,2,1],[0,1,2,8,2],[4,3,7,8,7],[6,9,8,9,3],[3,8,3,1,5]])
# A = np.array([[8,8,4,1,1,7],[1,4,8,5,8,1],[8,7,3,8,5,1],[4,1,1,9,9,1],[0,5,7,3,0,0],[3,1,1,0,0,0]])
A = np.array([[2,1,0],[1,3,1],[0,1,4]])
# row = 6
# col = 6
# A = np.random.randint(10,size=(row,col))
A1 = A
A2 = A1
row,col = A.shape
print(row)
print(A)


# In[3]:


# Initialize G list
G_lst = []
Q = np.eye(row)


# In[4]:


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

# A5 = A
# A5[5,5] = A5[5,5] * -1
print(A)
print(Q)    


# In[5]:


# # Initialize variables
# x = A[4,4]
# y = A[5,4]
# r = np.sqrt(x**2 + y**2)

# cos_t = x/r
# sin_t = y/r

# # Initialize Given Rotation Matrix
# G7 = np.eye(row)
# # # Setup given rotation matrix
# # rotation = np.array([[cos_t,sin_t],[-sin_t,cos_t]])
# # G6[2:5,2:5] = rotation

# G7 = np.eye(row)
# G7[4,4] = cos_t
# G7[5,4] = sin_t
# G7[4,5] = -sin_t
# G7[5,5] = cos_t
# print(G7)
# G_lst.append(G7)

# G_lst.append(G7)

# A7 = np.dot(G7,A)


# In[6]:


#Q
G_lst_len = len(G_lst)
G_lst = np.array(G_lst)
# Q_mat = np.dot(G1.T,np.dot(G2.T,G3.T))
Q_mat = G_lst[0].T
for i in range(1,G_lst_len):
#     print(Q_mat)
    Q_mat = Q_mat.dot(G_lst[i].T)
print(Q_mat)


# In[7]:


print(A)


# In[8]:


B = np.dot(Q_mat,A)
print(B)


# In[9]:


Z = A
# Z[5,5] = Z[5,5] * -1
# print(Z)
Y = np.dot(Q_mat,Z)
print(Y)


# In[10]:


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

 


# In[11]:


A = np.array([[2,1,0],[1,3,1],[0,1,4]])
Q,R = QR_Givens(A)
print(Q)
print(R)


# In[12]:


C = np.array([[2,1,0],[1,3,1],[0,1,4]])
Q,R = QR_Givens(C)


# In[13]:


A = np.array([[2,1,0],[1,3,1],[0,1,4]])
pQ = np.eye(A.shape[0])
X=A.copy()
for i in range(100):
        Q,R = QR_Givens(X)
        pQ = pQ.dot(Q)
        X = R.dot(Q)


# In[14]:


print(X)
print(pQ)


# In[15]:


w,v = np.linalg.eig(A)
print(w)
print(v)


# In[16]:


def QR_eig(A):
    pQ = np.eye(A.shape[0])
    X=A.copy()
    for i in range(100):
            Q,R = QR_Givens(X)
            pQ = pQ.dot(Q)
            X = R.dot(Q)
            
                
    return pQ,X


# In[17]:


Q,R = QR_eig(A)
print(Q)
print(R)

# Q,R = QR_Givens(A)
# print(Q)
# print(R)
# A2 = Q.dot(R)
# print(A2)


# In[18]:


Q,R = QR_Givens(A)
print(Q)
print(R)
AA = Q.dot(R)
print(AA)


# In[ ]:




