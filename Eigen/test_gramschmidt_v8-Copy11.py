#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[2]:


# X =  np.array([1,0,0,1,1,0,1,1,1]).reshape(3,3)
# X =  np.array([1,0,0,0,1,1,0,0,1,1,1,1,1,0,0,1]).reshape(4,4)
row = 6
col = 6
X = np.random.randint(2,size=(row,col))
Y = Z = X.copy()
X
# row,col = X.shape
# # print(X)
# Y = X.copy()
# print(X)

X = np.array([1,0,1,0,1,1,1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,1,0,1,0,1,1,1,0,1,1,1,0,0,0,1]).reshape(6,6)
Y = X.copy()

if row != col:
    print('Not Equal !')
    X = X.T.dot(X)
    row,col = X.shape
    Y = X.copy()
    print(row,col)

row,col = X.shape
print(X)
print(row,col)


# In[3]:


Q,R = np.linalg.qr(Y)
Q1 = Q.copy()

print(Q)
print('\n')
print(R)

print('\n')

X =Q.dot(R)
print(X)

print('\n')

# np.dot(Q[row-1],Q[row-1].T)


# In[4]:


# Initialize arrays

U = []
V_norm = []
U_Norm = []
A = []
projection = []


# In[5]:


# First Column

a1 = X[:,0]
A.append(a1)
u1 = a1
print(u1)
U.append(u1)

# Normalization
a1_norm = np.linalg.norm(a1)
u1_normalized = a1/a1_norm
print(u1_normalized)
V_norm.append(u1_normalized)
print(V_norm)

V_norm_1 = np.array(V_norm).T
vv = np.multiply(V_norm_1[0],Y)
print(vv)

# V_norm = np.array(V_norm)
# R1 = np.dot(V_norm,Y)
# print(R1)
# R = np.dot(V_norm[0],Y)[0]
# print(R)

# V_norm = np.array(V_norm)
# R = np.dot(V_norm,Y)[0]
# print(R)

# V_norm = np.array(V_norm)
# R1 = np.dot(V_norm,Y)
# print(R1)
# R = np.dot(V_norm[0],Y)[0]
# print(R)


# In[6]:


# Subsequent Columns
for j in range(1,row):
    a = X[:,j]
    u = a
    A.append(a)
    
    # Projection
    projection = []
    for i in range(j):

        # Dot products
        # Denominator
        denom = U[i-1].dot(U[i-1].T)
        # Numerator
        num = U[i-1].dot(a)
        # Multiple
        vec = np.multiply(num/denom,U[i-1])
        projection.append(vec)
        
#     print(projection)
    
    for k in range(len(projection)):
        u1 = a - projection[k]
        a = u1
#     print(a)
    
    U.append(u1)
    
    # Normalization
    v_norm = np.linalg.norm(u1)
    v_normalized = u1/v_norm
#     print(v_normalized)
    V_norm.append(v_normalized)
    
# V_norm = np.array(V_norm)
# R = np.dot(V_norm[1],Y)
# print(R)

V_norm = np.array(V_norm)
print('V norm:', V_norm)
R1 = np.dot(V_norm,Y)
print(R1)
R_0 = []
for i in range(1,row):
    print('i:',i)
    R = np.dot(V_norm[row-i],Y)[0]
    R_0.append(R)
print(R_0)

print('\n')
# U_norm = []
# for i in V_norm:
#     U_norm.append(list(i))
print(U)
print('\n')
print(V_norm)
# print(V_norm[0])
# print('\n')
# print(U)


# In[7]:


V_norm = np.array(V_norm)
print('V norm:', V_norm)
# R1 = np.dot(V_norm.T,Y)
# print(R1)
R_0 = []
for i in range(1,row):
    print('i:',i)
    R = np.dot(V_norm[row-i],Y)[0]
    R_0.append(R)
print(R_0)

for i in range(len(R_0)):
    if abs(R_0[i]) > 1e-2:
        a1 = X[:,0]
        A[0] = a1
        u1 = a1
        print(u1)
#         U.append(u1)
        U[0] = u1

        # Normalization
        a1_norm = np.linalg.norm(a1)
        u1_normalized = a1/a1_norm
        print(u1_normalized)
#         V_norm.append(u1_normalized)
        V_norm[0] = u1_normalized
        print(V_norm)


# In[8]:


# # Subsequent Columns
# for j in range(1,row):
#     a = X[:,j]
#     u = a
#     A.append(a)
    
#     # Projection
#     projection = []
#     for i in range(j):

#         # Dot products
#         # Denominator
#         denom = U[i-1].dot(U[i-1].T)
#         # Numerator
#         num = U[i-1].dot(a)
#         # Multiple
#         vec = np.multiply(num/denom,U[i-1])
#         projection.append(vec)
        
# #     print(projection)
    
#     for k in range(len(projection)):
#         u1 = a - projection[k]
#         a = u1
# #     print(a)
    
#     U.append(u1)
    
#     # Normalization
#     v_norm = np.linalg.norm(u1)
#     v_normalized = u1/v_norm
# #     print(v_normalized)
#     V_norm.append(v_normalized)
    
# # V_norm = np.array(V_norm)
# # R = np.dot(V_norm[1],Y)
# # print(R)

# V_norm = np.array(V_norm)
# print('V norm:', V_norm)
# R1 = np.dot(V_norm,Y)
# print(R1)
# R_0 = []
# for i in range(1,row):
#     print('i:',i)
#     R = np.dot(V_norm[row-i],Y)[0]
#     R_0.append(R)
# print(R_0)

# print('\n')
# # U_norm = []
# # for i in V_norm:
# #     U_norm.append(list(i))
# print(U)
# print('\n')
# print(V_norm)
# # print(V_norm[0])
# # print('\n')
# # print(U)


# In[9]:


# V_norm = np.array(V_norm)
# R = np.dot(V_norm[1],Y)[0]
# #     print(R)


# In[10]:


err = []
for i in range(len(U)):
    err.append(np.dot(U[i],U[i].T))
    
print(err)

# print(V_norm[3].T.dot(V_norm[3]))


# In[11]:


# for i in range(row):
#     Q1[i,:] = np.multiply(-1,Q1[i,:])
# print(Q1)
# print('\n')
# print(err)
# print('\n')
# print(V_norm)


# In[12]:


V_norm = np.array(V_norm)
R = np.dot(V_norm[1],Y)
print(R)

# V_norm_0 = V_norm[0].T.dot(Y)
# print(V_norm_0)

V_norm_1 = V_norm.T
print("V_norm_1:", V_norm_1)
R = np.dot(V_norm_1,Y)
R1 = np.dot(V_norm_1[0],Y)
print(R)
print(R1)


# In[13]:


def QR2(Y):
    for _ in range(10):
    # Initialize arrays

        U = []
        V_norm = []
        U_Norm = []
        A = []
        projection = []
        Z = Y.copy()

        # First Column

        a1 = Y[:,0]
        A.append(a1)
        u1 = a1
        print(u1)
        U.append(u1)

        # Normalization
        a1_norm = np.linalg.norm(a1)
        u1_normalized = a1/a1_norm
        print(u1_normalized)
        V_norm.append(u1_normalized)
        
            
        print('u1_normalized',u1_normalized)
        
        for m in range(1,len(u1_normalized)):
#             print("m:", u1_normalized[m])
            if abs(u1_normalized[m]) < 0 or u1_normalized[m] == nan:
               # Normalization
            
                a1 = Y[:,0:1]
                A[0] = a1
                u1 = a1
                print(u1)
#                 U.append(u1)
                U[0] = u1
            
                a1_norm = np.linalg.norm(a1)
                u1_normalized = a1/(a1_norm + 1e-15)
                print(u1_normalized)
#                 V_norm.append(u1_normalized)
                print('mistake!')
                V_norm[0] = u1_normalized
        
#         print('u1_normalized',u1_normalized)

        # Subsequent Columns
        for j in range(1,row):
            a = Y[:,j]
            u = a
            A.append(a)

            # Projection
            projection = []
            for i in range(j):

                # Dot products
                # Denominator
                denom = (U[i-1].dot(U[i-1].T) + 1e-15)
                # Numerator
                num = U[i-1].dot(a)
                # Multiple
                vec = np.multiply(num/denom,U[i-1])
                projection.append(vec)

        #     print(projection)

            for k in range(len(projection)):
                u1 = a - projection[k]
                a = u1
            print("a_0:",a)

            U.append(u1)

            # Normalization
            v_norm = np.linalg.norm(u1)
            v_normalized = u1/(v_norm + 1e-15)
        #     print(v_normalized)
            V_norm.append(v_normalized)

        print('\n')
        # U_norm = []
        # for i in V_norm:
        #     U_norm.append(list(i))
        print(U)
        print('\n')
        print(V_norm)
        # print(V_norm[0])
        # print('\n')
        # print(U)

        V_norm = np.array(V_norm)
        R = np.dot(V_norm,Z)
        V_norm_1 = V_norm
        print('V_norm_1:',V_norm_1)
        R1 = np.dot(V_norm_1[0],Z)
    
    return V_norm_1,V_norm.T,R,R1


# In[14]:



V,Q,R,R1 = QR2(Z)
# print(Q)
# print('\n')
# print(R)
# print('\n')
# print(R1)


# In[15]:


print(V)
print('\n')
print(Q)
print('\n')
print(R)
print('\n')
print(R1)


# In[ ]:




