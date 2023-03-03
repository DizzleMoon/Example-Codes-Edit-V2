#!/usr/bin/env python
# coding: utf-8

# In[49]:


# !pip install import-ipynb
import import_ipynb

import sympy as sy
import numpy as np
from sympy.functions import sin,cos,exp
import matplotlib.pyplot as plt
from sympy import* 
from test_gcf__v2 import*
import math
import numpy as np
import math
import fractions
import copy
import decimal

plt.style.use("ggplot")


# Note to users.
# Code was designed for centre differences of fourth order accuracy and, possibly, higher.
# 
# Fourth Order centre difference fomulas begin with system matrix and solution vector with a size of 7 or higher.

# In[50]:


# Centre Difference
# Create System Matrix and Solution Vector to calculate weights

# Create System Matrix
def sys_matrix(n):
    if n < 5:
        print("Matrix size must be 7 or more.")
        return
    
    # Create list of zeros
    if n%2 == 0:
     # Create matrix by breaking it into 2 halves
        m = int(n/2)
        mat_list_1 = np.zeros((n,m))
        mat_list_2 = np.zeros((n,m))

        # First Half
        for i in reversed(range(m)):
            for j in range(n):
                mat_list_1[j,i] = ((m-i)**j)/math.factorial(j)

        # Second Half
        for i in range(m):
            for j in range(n):
                mat_list_2[j,i] = ((m-i)**j)*(-1)**j/math.factorial(j)

        # Concatenate 
        mat_list = np.hstack((mat_list_1,np.flip(mat_list_2,1)))
        print(mat_list)
    else:
        mat_list = np.matrix(np.zeros((n,n)))
        print(mat_list)
        m = floor(n/2)
        print(m)
        # First half
        print(n)
        for i in range(m+1):
            for j in range(n):
                mat_list[j,i] = ((m-i)**j)
                mat_list[j,i + m] = ((i)**j)*(-1)**j

        # Middle function
        mat_list[0,m] = 1
        print(mat_list)

    return mat_list


# In[51]:


# Determine derivative level
derivative = 4
index = derivative
#System size
n = 7
if derivative < 3:
    n = n - 2
# Create System matrix
mat_list = sys_matrix(n)
print(mat_list)


# In[52]:


# Create System matrix
mat_list = sys_matrix(n)
# Determine derivative level
# derivative = 1
# index = derivative
# Initialize solution vector
b = np.zeros((1,n))
# Input varaibles into solution vector
b[0][index] = derivative
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)
# Flag to ensure weights are of correct polarity
rem = derivative%2
sign = 1
if rem == 0:
    sign = -1
c = c/c[0] * sign
print(c)


# In[53]:


# derivative%2
print(sign)


# In[54]:


# Sine Function
f = lambda x: 1/(x**2)
dx = 0.01
t = -1

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(dx**4)

# Derivative coefficient
deriv = (np.matmul(func,c))/math.factorial(4)
print(deriv)


# In[55]:


# # Sine Function
# f = lambda x: 1/(x**2)
# dx = 0.01
# t = 2

# # Function matrix
# func = np.matrix([f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx)])/(dx**1)
# print(func)

# # Derivative coefficient
# deriv = (np.matmul(func,c))/math.factorial(1)
# print(deriv)


# In[56]:


# Create Function
def taylor_series(derivative,n):
    
    # For fourth order CD, size of matrix must be 7 or higher
    if n < 7:
        print("Matrix size must be 7 or more.")
        return
    
    # Determine derivative level
    index = derivative
    #System size
    n = 7
    if derivative < 3:
        n = n - 2
    # Create System matrix
    mat_list = sys_matrix(n)
    print(mat_list)
    
   
    # Create list of zeros
    if n%2 == 0:
     # Create matrix by breaking it into 2 halves
        m = int(n/2)
        mat_list_1 = np.zeros((n,m))
        mat_list_2 = np.zeros((n,m))

        # First Half
        for i in reversed(range(m)):
            for j in range(n):
                mat_list_1[j,i] = ((m-i)**j)/math.factorial(j)

        # Second Half
        for i in range(m):
            for j in range(n):
                mat_list_2[j,i] = ((m-i)**j)*(-1)**j/math.factorial(j)

        # Concatenate 
        mat_list = np.hstack((mat_list_1,np.flip(mat_list_2,1)))
        print(mat_list)
    else:
        mat_list = np.matrix(np.zeros((n,n)))
        print(mat_list)
        m = floor(n/2)
        print(m)
        # First half
        print(n)
        for i in range(m+1):
            for j in range(n):
                mat_list[j,i] = ((m-i)**j)
                mat_list[j,i + m] = ((i)**j)*(-1)**j

        # Middle function
        mat_list[0,m] = 1
        print(mat_list)
        
    # Create System matrix
    mat_list = sys_matrix(n)
    # Initialize solution vector
    b = np.zeros((1,n))
    # Input varaibles into solution vector
    b[0][index] = derivative
    print(b)
    # Weights
    c = np.linalg.pinv(mat_list)*b.T
    print(c)
    
    print(c[0])
    den = 1/c[0]
    print("Denominator:",den)
    
    # Flag to ensure weights are of correct polarity
    rem = derivative%2
    sign = 1
    if rem == 0:
        sign = -1
    c = c/c[0] * sign
#     c = c * sign
    print(c)
   
    return c,den


# In[57]:


# Derivative
derivative = 4
# Order
n = 7
c,den = taylor_series(derivative,n)
print(c)
print(list(den))


# In[58]:


# # Function matrix
# dx = 0.0001
# t = 1
# f = lambda x: 1/(x**2)
# func = np.matrix([f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx)])/(dx**derivative)
# deriv = np.matmul(func,c)/math.factorial(1)
# print(deriv)


# In[59]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 7
# Delta
dx = 0.0001
# Guess
t= dx*100
# Sine Function
f = lambda x: np.sin(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(6*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = (np.matmul(func,c)/math.factorial(term)) * 2


print(deriv)


# In[60]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 6
# Delta
dx = 0.0001
# Guess
t= dx/100
# Sine Function
f = lambda x: np.cos(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(36*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = (np.matmul(func,c)/math.factorial(term))*2*dx*100


print(deriv)


# In[61]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 6
# Delta
dx = 0.00001
# Guess
t= dx/100
# Sine Function
f = lambda x: np.exp(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(36*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = (np.matmul(func,c)/math.factorial(term))*2*dx/10


print(deriv)


# In[62]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 3
# Delta
dx = 0.001*term
# Guess
t= 0
# Sine Function
f = lambda x: np.log(1+x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(36*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = np.matmul(func,c)/term


print(deriv)


# In[63]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 7
# Delta
dx = 0.01
# Guess
t= 0
# Sine Function
f = lambda x: 1/(1-x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(36*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = np.matmul(func,c)/4


print(deriv)


# In[64]:


# Trigonometry
# Cosh
# Derivative Coefficient Term: Must be EVEN
term = 7
# Delta
dx = 0.01
# Guess
t= 0
# Sine Function
f = lambda x: 1/(1+x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(36*dx**4)

# Sign
# sign = 2
# rem = term%4
# if rem == 3:
#     sign = 1  

# Derivative coefficient
deriv = np.matmul(func,c)/4


print(deriv)


# In[65]:


var( 'x' )
formula = 1/(1+x)
series(formula, x, 0, 8).removeO()


# In[ ]:




