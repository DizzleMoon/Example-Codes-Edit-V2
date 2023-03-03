#!/usr/bin/env python
# coding: utf-8

# In[71]:


import sympy as sym
import numpy as num
import matplotlib.pyplot as pyplot

import sympy as sy
import numpy as np
from sympy.functions import sin,cos,exp
import matplotlib.pyplot as plt
from sympy import* 
import math
plt.style.use("ggplot")

pyplot.style.use("ggplot")


# In[72]:


def centre_difference(order):
    # Initialize variables
    if order == "first-order":
        # First Order
        # System Matrix
        n = 4
        # Solution
        # Value
        sol_val = 12
        # index 
        index = 1 
        
    elif order == "second-order":
        # Second Order
        # System Matrix
        n = 5
        # Solution
        # Value
        sol_val = 12
        # index 
        index = 2
        
    elif order == "third-order":
        # Third Order
        # System Matrix
        n = 6
        # Solution
        # Value
        sol_val = 8
        # index 
        index = 3
        
    elif order == "fourth-order":
        # Fourth Order
        # System Matrix
        n = 7
        # Solution
        # Value
        sol_val = 6
        # index 
        index = 4
        
    return n,sol_val,index


# In[73]:


# Function
n,sol_val,index = centre_difference("fourth-order")


# In[74]:


# Create System Matrix
def sys_matrix(n):
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
        m = floor(n/2)

        for i in range(m+1):
            for j in range(n):
                mat_list[j,i] = ((m-i)**j)/math.factorial(j)
                mat_list[j,i + m] = ((i)**j)*(-1)**j/math.factorial(j)

        # Middle function
        mat_list[0,m] = 1
        print(mat_list)
        
    return mat_list


# In[75]:


# Create System Matrix
mat_list = sys_matrix(n)
# Solution Vector
b = np.zeros((1,n))
# Index
b[0][index] = sol_val
# Create Weights
c = np.linalg.pinv(np.matrix(mat_list))*b.T
print(c)


# In[76]:


# Trigonometry
# Sine 
# Delta
dx = 0.0001
# Guess
t= dx*100
# Derivative Coefficient Term: Must be ODD
term = 13
# Sine Function
f = lambda x: np.sin(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(dx**4)

# Sign
sign = 2
rem = term%4
if rem == 3:
    sign = 1  

# Derivative coefficient
deriv = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (sign)

print(deriv)


# In[77]:


from sympy import*
var( 'x' )
formula = sin(x)
series(formula, x, 0, 18).removeO()


# In[78]:


# Trigonometry 
# Cosine 
# Delta
dx = 0.06001
# Guess
t= dx/100
# Derivative Coefficient Term: Must be EVEN
term = 14
# Sine Function
f = lambda x: np.cos(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(dx**4)

# Sign
sign = 1
rem = term%4
if rem == 0 or rem == 3:
    sign = 2  

# Derivative coefficient
deriv = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (sign)

print(deriv)


# In[79]:


from sympy import*
var( 'x' )
formula = cos(x)
series(formula, x, 0, 18).removeO()


# In[ ]:




