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
import warnings
import sys
from numpy import linalg as LA
from sympy import * 


# In[2]:


# def fxn():
#     warnings.warn("ignore", RuntimeWarning)


# In[3]:


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
#         eig_val = np.diag(X)
        
    return X,eig_vec


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
    
    


# In[4]:


# def power_method(A):
    
#     a = A.copy()
    
#     # Create eigenvalues & eigenvectors
#     eig_val,eig_vec= eigenspace(A)
    
#     # Matrix shape
#     row,col = A.shape
    
#     # Initialize variables
#     # Eigenvector
#     U = eig_vec
#     # Eigenvalue
# #     Lambda = np.abs(np.diag(eig_val))
#     Lambda = eig_val

#     # Tolerance
#     tol = 1e-3
#     # Magnitude
#     y0 = 0
#     # Index value for while loop
#     i = 0
    
#     # Guess
#     x0 = np.ones((1,row))
    
#     while tol > 1e-10:
        
#     # Determine eigenvector
#         y = U.dot(Lambda ** i).dot(np.linalg.pinv(U)).dot(x0.T)

#     # Normalize eigenvectors
#         y = y/y.max()
#         if row < 2:
#             y = y/y.min()
        
#     # Determine tolerance
#         tol = np.abs(np.linalg.norm(y) - y0)
        
#     # Update magnitude
#         y0 = np.linalg.norm(y)
        
#     # Update counter
#         i += 1
        
#     # Rayleigh_Quotient
#     x = y    
#     # Numerator
#     num = A.dot(x).dot(x.T)
#     num = np.linalg.pinv(A).dot(x).dot(x.T)
#     num = x.T.dot(a).dot(x)
#     # Denominator
#     den = np.dot(x.T,x)
#     # Final solution
#     rayleigh_quotient = num/den
# #     print('I:',i)
    
#     return rayleigh_quotient, y
    
# # A = np.array([[0,2],[2,3]])
# # x0 = np.array([[1,1]])

# def inverse_power_method(A):

#     # Eigenvalue, Eigenvector
#     rayleigh_quotient,y = power_method(A)

#     # Inverse eigenvalue
#     inverse_lambda = 1/rayleigh_quotient
#     print(inverse_lambda)

#     return inverse_lambda,y/y.max()


# In[5]:


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
    #     Lambda = np.abs(np.diag(eig_val))
    Lambda = eig_val

    # Tolerance
    tol = 1e-3
    # Magnitude
    y0 = 0
    # Index value for while loop
    i = 0

    # Guess
    x0 = np.ones((1,row))
    y1 = np.ones((1,row))

    while tol > 1e-10:

    # Determine eigenvector
        y = U.dot(Lambda ** i).dot(np.linalg.pinv(U)).dot(x0.T)
        if not sys.warnoptions:
            warnings.simplefilter("ignore")
            
        if np.isnan(np.linalg.norm(y)):
            y = y1
            break

    # Normalize eigenvectors
        y = y/y.max()
        y1 = y
        if row < 2:
            y = y/y.min()

    # Determine tolerance
        tol = np.abs(np.linalg.norm(y) - y0)

    # Update magnitude
        y0 = np.linalg.norm(y)

    # Update counter
        i += 1
        if i > 500:
            break

    print(i)

    # Rayleigh_Quotient
    x = y    
    # Numerator
    # num = a.dot(x).dot(x.T)
    # num = np.linalg.pinv(A).dot(x).dot(x.T)
    num = x.T.dot(a).dot(x)
    # num = np.dot(num.T,num)
    print(num)
    # Denominator
    den = np.dot(x.T,x)
    print(den)
    # Final solution
    rayleigh_quotient = num/den
    #     print('I:',i)
    print(rayleigh_quotient)
    
    return rayleigh_quotient, y
    
# A = np.array([[0,2],[2,3]])
# x0 = np.array([[1,1]])

def inverse_power_method(A):

    # Eigenvalue, Eigenvector
    rayleigh_quotient,y = power_method(A)

    # Inverse eigenvalue
    inverse_lambda = 1/rayleigh_quotient
    print(inverse_lambda)

    return inverse_lambda,y/y.max()


# In[6]:


# A = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
# alpha_1 = 2.1

# A = np.array([[1,2,0],[-2,1,2],[1,3,1]])
# x0 = np.array([[1,1,1]])

# A = np.array([[2,-12],[1,-5]])
# x0 = np.array([[1,1]])

A = np.array([[0,2],[2,3]])
x0 = np.array([[1,1]])

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


# In[7]:


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


# In[8]:


eig_val_inv,eig_vec_inv = inverse_power_method(a)
print('\n')
print('Inverse Power:',eig_val_inv)
print(eig_vec_inv)

eig_val_pow,eig_vec_pow = power_method(a)
print('\n')
print('Power Method:',eig_val_pow)
print(eig_vec_pow)
print('\n')


# In[9]:


# a = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])
# alpha = 4.2

print(a)

row,col = a.shape
# print(row)
# print(col)

w,v = eigenspace(a)
print(np.diag(w))
print(v)

eig_val_inv,eig_vec_inv = inverse_power_method(a)
print('\n')
print('Inverse Power:',eig_val_inv)
print(eig_vec_inv)

eig_val_pow,eig_vec_pow = power_method(a)
print('\n')
print('Power Method:',eig_val_pow)
print(eig_vec_pow)
print('\n')

constants = np.zeros((1,row))
constants[0][0] = eig_val_inv
constants[0][row-1] = eig_val_pow
# print(constants)

# if row > 3:
#     for i in range(1,row-1):
#     #     print(i)
#         print(w[i])
#         constants[0][i] = np.abs(w[i] - eig_val_pow)
#     #     print(constants)
#     #     print(eig_val_pow)
    
# alpha = 0
# alpha = eig_val_inv
# alpha = np.abs(w[1]-eig_val_pow)
alpha = eig_val_pow
# alpha = np.abs(eig_val_inv)
# print('alpha:',alpha)
# print('\n')
# print('Constants:',constants[0])

# ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,constants[0][row-1])
# print(ray_quo)
# print(eig_vec)
# print(eig_val)

# ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,alpha)
# print(ray_quo)
# print(eig_vec)
# print(eig_val)

print('Constants:',constants[0])

constants[0][0] = eig_val_inv
constants[0][row-1] = eig_val_pow

w1 = np.diag(w)
print('Eigenvalues:',w1)

if row > 2:
    for i in range(1,row-1):
    #     print(i)
        print(w[i])
        constants[0][i] = np.abs(w1[i] - eig_val_inv)
    #     print(constants)
    #     print(eig_val_pow)
    
    print('Constants:',constants)

    for i in range(1,row-1):
        print(constants[0][i])
        ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,constants[0][i])
        print(ray_quo)
        print(eig_vec)
        print(eig_val)
        constants[0][i] = eig_val
        
        
print(constants)

# print(constants[0][1])
# b = constants[0][1]
# print(b)

# ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,b)
# print(ray_quo)
# print(eig_vec)
# print(eig_val)


print('\n')

alpha = 4.2
ray_quo,eig_vec,eig_val = shift_inverse_power_method(a,alpha)
print(ray_quo)
print(eig_vec)
print(eig_val)


# In[10]:


w,v = eigenspace(a)
print(np.diag(w))
print(v)
A = v.dot(w).dot(np.linalg.inv(v))
print(A)
print(a)


# In[11]:


# # a = A.copy()
# a = np.array([[0,11,-5],[-2,17,-7],[-4,26,-10]])

# # Create eigenvalues & eigenvectors
# eig_val,eig_vec= eigenspace(A)

# # Matrix shape
# row,col = A.shape

# # Initialize variables
# # Eigenvector
# U = eig_vec
# # Eigenvalue
# #     Lambda = np.abs(np.diag(eig_val))
# Lambda = eig_val

# # Tolerance
# tol = 1e-3
# # Magnitude
# y0 = 0
# # Index value for while loop
# i = 0

# # Guess
# x0 = np.ones((1,row))
# y1 = np.ones((1,row))

# while tol > 1e-10:

# # Determine eigenvector
#     y = U.dot(Lambda ** i).dot(np.linalg.pinv(U)).dot(x0.T)
#     if np.isnan(np.linalg.norm(y)):
#         y = y1
#         break

# # Normalize eigenvectors
#     y = y/y.max()
#     y1 = y
#     if row < 2:
#         y = y/y.min()

# # Determine tolerance
#     tol = np.abs(np.linalg.norm(y) - y0)
    
# # Update magnitude
#     y0 = np.linalg.norm(y)
    
# # Update counter
#     i += 1
#     if i > 500:
#         break
        
# print(i)

# # Rayleigh_Quotient
# x = y    
# # Numerator
# # num = a.dot(x).dot(x.T)
# # num = np.linalg.pinv(A).dot(x).dot(x.T)
# num = x.T.dot(a).dot(x)
# # num = np.dot(num.T,num)
# print(num)
# # Denominator
# den = np.dot(x.T,x)
# print(den)
# # Final solution
# rayleigh_quotient = num/den
# #     print('I:',i)
# print(rayleigh_quotient)


# In[12]:


print(a,'\n')

def normalize(x):
    fac = abs(x).max()
    x_n = x / x.max()
    return fac, x_n

a_inv = np.linalg.inv(a)

x = np.array([1, 1])

for i in range(18):
    x = np.dot(a_inv, x)
    lambda_1, x = normalize(x)
    
print('Eigenvalue:', lambda_1)
print('Eigenvector:', x)
print('\n')

w,v = eigenspace(a)
print(np.diag(w))
print(v)

