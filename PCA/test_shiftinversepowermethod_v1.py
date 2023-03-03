#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
# np.set_printoptions(linewidth=np.inf)
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
        
    return eig_val,eig_vec


def CoV(df):
    
    # Shape of dataframe
    row,col = df.shape
    print(row,col)
    
    # Determine indices
#     if row > col:
#         row = col
#         col = row
    
    # Initialize Covariance matrix
    CoVar = np.zeros((col,col))
    
    # List of means
    stats = []
    for i in range(col):
        stats.append(df.iloc[:,i].mean())

    # Solve covariance matrix        
    for k in range(col):
        for i in range(col):
            var = 0
            for j in range(row):
                var = var + ((df.iloc[j,k] - stats[k])*(df.iloc[j,i] - stats[i]))/(row-1)
            CoVar[k][i] = var
        
    return CoVar
    
    


# In[3]:


def power_method(A):
    
    a = A.copy()
    
    # Create eigenvalues & eigenvectors
    eig_val,eig_vec= eigenspace(A)
    
    # Matrix shape
    row,col = A.shape
    
    # Initialize variables
    # Eigenvector
    U = eig_vec
    # Eigenvalue
    Lambda = np.abs(np.diag(eig_val))

    # Tolerance
    tol = 1e-3
    # Magnitude
    y0 = 0
    # Index value for while loop
    i = 0
    
    # Guess
    x0 = np.ones((1,row))
    
    while tol > 1e-15:
        
    # Determine eigenvector
        y = U.dot(Lambda ** i).dot(np.linalg.pinv(U)).dot(x0.T)

    # Normalize eigenvectors
#         y = y
#         y = 1/y
        y = y/y.max()
        if row < 2:
            y = y/y.min()
#         y = y[::-1]/y.max()
#         y = y/y.min()
#         print(y)
        
    # Determine tolerance
        tol = np.abs(np.linalg.norm(y) - y0)
        
    # Update magnitude
        y0 = np.linalg.norm(y)
        
    # Update counter
        i += 1
        
    # Rayleigh_Quotient
#     x = np.array([1,1])
    x = y

    num = A.dot(x).dot(x.T)
    num = np.linalg.pinv(A).dot(x).dot(x.T)
    num = x.T.dot(a).dot(x)
    print(num)
    den = np.dot(x.T,x)
    print(den)
    rayleigh_quotient = num/den
    print('I:',i)
    
    return rayleigh_quotient, y
    
# A = np.array([[0,2],[2,3]])
# x0 = np.array([[1,1]])

def inverse_power_method(A):

#     # Initial eigenvalues
#     eig_val,eig_vec= eigenspace(A)
# #     print(eig_val)
# #     print(eig_vec)
    
#     # Matrix shape
#     row,col = A.shape

#     iter = 10
#     U = eig_vec
#     print(U)
#     Lambda = np.abs(np.diag(eig_val))
#     # Lambda = np.abs(Lambda)
#     print(Lambda)
#     print('\n')
#     tol = 1e-3
#     y0 = 0
#     i = 0
    
#     # Guess
#     x0 = np.ones((1,row))
    
#     while tol > 1e-15:
#         y = U.dot(Lambda ** i).dot(np.linalg.pinv(U)).dot(x0.T)
#     #     y = np.linalg.pinv(A).dot(x0.T)
#     #     x0 = y.T
#     #     y = np.linalg.pinv(U).dot(np.linalg.pinv(Lambda ** i)).dot(U).dot(x0.T)
#         # Inverse
#     #     y_inv = U.dot(np.linalg.pinv(Lambda ** i)).dot(np.linalg.pinv(U))
#     #     y = y_inv.dot(x0.T)
#     #     print('Lambda:',Lambda**i)
#         y = y/y.max()
#         if row < 3:
#             y = y/y.min()
#     #     x0 = y.T
# #         print(np.linalg.norm(y))
#         tol = np.abs(np.linalg.norm(y) - y0)
#         y0 = np.linalg.norm(y)
#         i += 1
        
#     # Rayleigh_Quotient
#     # x = np.array([1,1])
#     x = y

#     # num = A.dot(x).dot(x.T)
#     # num = np.linalg.pinv(A).dot(x).dot(x.T)
#     num = x.T.dot(A).dot(x)
#     print(num)
#     den = np.dot(x.T,x)
#     print(den)
#     rayleigh_quotient = num/den

#     # rayleigh_quotient = A.dot(x).dot(x)/np.dot(x,x)
#     print(rayleigh_quotient)

    rayleigh_quotient,y = power_method(A)

    # Inverse eigenvalue
    inverse_lambda = 1/rayleigh_quotient
    print(inverse_lambda)

    return inverse_lambda,y/y.max()


# In[4]:


# A = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
# alpha_1 = 2.1

# A = np.array([[1,2,0],[-2,1,2],[1,3,1]])
# x0 = np.array([[1,1,1]])

A = np.array([[2,-12],[1,-5]])
x0 = np.array([[1,1]])

# A = np.array([[0,2],[2,3]])
# x0 = np.array([[1,1]])

# A = np.array([[0,1],[1,1]])
# x0 = np.array([[1,1]])

# A = np.array([[2,-3,-1],[7,-6,-1],[-16,14,3]])
# x0 = np.array([[1,1,1]])

# A = np.array([[6,-2,2,4],[-2,3,-1,3],[2,-1,3,-10],[7,-2,5,8]])
# x0 = np.array([[1,1,1,1]])

# A = np.array([[3,-1,0],[-2,4,-3],[0,-1,1]])
# x0 = np.array([[1,1,1]])

# A = np.array([[8,-5],[-6,7]])
# x0 = np.array([[1,1]])

# A = np.array([[6,-8,2,4],[-2,9,-1,3],[2,-1,3,-8],[7,-2,4,9]])
# x0 = np.array([[1,1,1,1]])

# A = np.array([[5,1],[4,2]])
# x0 = np.array([[1,1]])


a = A.copy()
print(a)


# In[5]:


eig_val_inv,eig_vec_inv = inverse_power_method(A)
print('\n')
print('Inverse Power:',eig_val_inv)
print(eig_vec_inv)

eig_val_pow,eig_vec_pow = power_method(A)
print('\n')
print('Power Method:',eig_val_pow)
print(eig_vec_pow)
print('\n')


# In[6]:


# B = A.copy()
# # Size of B
# row,col = B.shape

# # First Guess
# alpha_1 = 0.5
# B_1 = B - alpha_1 * np.eye(row)

# c = []
# X = []

# # X vector
# Y = np.ones((row,1))

# # Y
# Y_0 = np.linalg.inv(B_1).dot(Y)
# #     print(Y_0)

# # Normalize
# C = Y_0/np.max(Y_0)
# #     print(C)
# #     print('\n')

# #     for _ in range(10):
# #         Y_0 = np.linalg.inv(B_1).dot(C)
# #         X.append(np.min(Y_0))
# #         # Normalize
# #         C = Y_0/np.min(Y_0)
# #         c.append(C)

# tol = 1e-3
# Y_1 = 0
# while tol > 1e-10:

#     Y_0 = np.linalg.inv(B_1).dot(C)
#     tol = np.abs(np.max(Y_0) - Y_1)
# #         print(tol)
#     Y_1 = np.min(Y_0)
#     X.append(np.max(Y_0))
#     # Normalize
#     C = Y_0/np.max(Y_0)
#     c.append(C)
    
# # Eigenvalues from Shift Inverse Power Method
# eig = 1/X[-1] + alpha_1
# print(eig)

# # print(c)
# # print(X)


# In[7]:


# # Eigenvalues from Shift Inverse Power Method
# eig = 1/X[-1] + alpha_1
# print(eig)

# w,v = eigenspace(B)
# print(w)
# print(v)


# In[8]:


A = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
alpha = 2.1

w,v = eigenspace(A)
print(w)
print(v)


# In[9]:


row,col = A.shape


X0 = np.ones((row,1))

shift = A - alpha*np.eye(row)

Yk = np.linalg.inv(shift).dot(X0)
print(Yk)

num_c = Yk.T.dot(X0)
den_c = X0.T.dot(X0)
c = num_c/den_c
print(c)

Xk = Yk/np.min(Yk)
print(Xk)

X0 = Xk

Yk = np.linalg.inv(shift).dot(X0)
print(Yk)

num_c = Yk.T.dot(X0)
den_c = X0.T.dot(X0)
c = num_c/den_c
print(c)

Xk = Yk/np.min(Yk)
print(Xk)

X0 = Xk

Yk = np.linalg.inv(shift).dot(X0)
print(Yk)

num_c = Yk.T.dot(X0)
den_c = X0.T.dot(X0)
c = num_c/den_c
print(c)

Xk = Yk/np.min(Yk)
print(Xk)


# In[10]:



# # A = np.array([[1,2,0],[-2,1,2],[1,3,1]])
# # x0 = np.array([[1,1,1]])

# A = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
# alpha = 4.2

# # Initialize Arrays
# # Shape of Matrix
# row,col = A.shape
# # Guess Vector
# X0 = np.ones((row,1))

# # Initialize constant
# alpha = 4.2

# # Shift Equation
# shift = A - alpha*np.eye(row)

# # Intialize variables
# # Tolerance
# tol = 1e-2
# # Rayleigh Quotient
# c0 = 0
# # Loop counter
# i = 0

# while tol > 1e-5:
    
#     # Shift inverse power method equation
#     Yk = np.linalg.inv(shift).dot(X0)
    
#     # Rayleigh Quotient: eigenvalue
#     num_c = Yk.T.dot(X0)
#     den_c = X0.T.dot(X0)
#     c = num_c/den_c
    
#     # Update guess: eigenvector
#     Xk = Yk/np.min(Yk)
    
#     # Tolerance
#     tol = np.abs(c0 - c)
    
#     # Update variables
#     # Guess vector
#     X0 = Xk
#     # Rayleigh Quotient
#     c0 = c
#     # Loop counter
#     i += 1
#     if i > 1000:
#         break
    
# # Normalized eigenvector    
# Xk_norm = Xk/Xk.max()

# # print(i)
# print(c)
# print(Xk)
# print(Xk_norm)
# print('\n')

# # Final estimated Eigenvalues from Shift Inverse Power Method
# eig = 1/c + alpha
# print(eig)
# print('\n')

# # Eigenvalue & Eigenvector
# # w,v = eigenspace(A)
# w,v = np.linalg.eig(A)
# print(w)
# print(v)


# In[11]:


def shift_inverse_power_method(A,alpha):
    
    # Initialize Arrays
    # Shape of Matrix
    row,col = A.shape
    # Guess Vector
    X0 = np.ones((row,1))

#     # Initialize constant
#     alpha = 4.2

    # Shift Equation
    shift = A - alpha*np.eye(row)

    # Intialize variables
    # Tolerance
    tol = 1e-2
    # Rayleigh Quotient
    c0 = 0
    # Loop counter
    i = 0

    while tol > 1e-15:

        # Shift inverse power method equation
        Yk = np.linalg.inv(shift).dot(X0)

        # Rayleigh Quotient: eigenvalue
        num_c = Yk.T.dot(X0)
        den_c = X0.T.dot(X0)
        c = num_c/den_c

        # Update guess: eigenvector
        Xk = Yk/np.min(Yk)

        # Tolerance
        tol = np.abs(c0 - c)

        # Update variables
        # Guess vector
        X0 = Xk
        # Rayleigh Quotient
        c0 = c
        # Loop counter
        i += 1
        if i > 1000:
            break

    # Normalized eigenvector    
    Xk_norm = Xk/Xk.max()

    # print(i)
#     print(c)
#     print(Xk)
#     print(Xk_norm)
#     print('\n')

    # Final estimated Eigenvalues from Shift Inverse Power Method
    eig = 1/c + alpha
#     print(eig)
#     print('\n')
    
    return c,Xk_norm,eig


# In[12]:


eig_val_inv,eig_vec_inv = inverse_power_method(a)
print('\n')
print('Inverse Power:',eig_val_inv)
print(eig_vec_inv)

eig_val_pow,eig_vec_pow = power_method(a)
print('\n')
print('Power Method:',eig_val_pow)
print(eig_vec_pow)
print('\n')


# In[13]:


a = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
alpha = 4.2

# alpha = eig_val_inv
alpha = np.abs(eig_val_pow-eig_val_inv)
print(alpha)
print('\n')

ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,alpha)
print(ray_quo)
print(eig_vec)
print(eig_val)

print('\n')

w,v = eigenspace(a)
print(w)
print(v)


# In[ ]:




