#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


x = 0
h = 0.001
n = 4
term = 3
# f = lambda x: np.exp(x)


# In[3]:


# Create list of zeros
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j

# Middle function
mat_list[0,m] = 1
print(mat_list)


# In[4]:


for i in range(0,4):
    print(1/math.factorial(i+1)*(-1)**(i+1))


# In[5]:


# Create list of zeros
n = 6
mat_list = np.matrix(np.zeros((n,n)))
print(mat_list)
# m = floor(n/2)
m = 2
print(m)
for j in range(m): 
    for i in range(1,n+1):
        mat_list[i-1,3*j] = ((m**(i))/math.factorial(i)) 
        mat_list[i-1,3*j-2]=(1/math.factorial(i+1))
        if j > (m/2 - 1):
            mat_list[i-1,3*j] = ((m**(i))/math.factorial(i))*(-1)**(i+1)
            mat_list[i-1,3*j-1]=(1/math.factorial(i+1)) * (-1)**(i+1)            
print(mat_list.T)


# In[6]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
b[0][term-1] = math.factorial(term-1)
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[7]:


# Create list of zeros
n = 5
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m+1):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j/math.factorial(j)

# Middle function
mat_list[0,m] = 1
print(mat_list)


# In[8]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][4] = math.factorial(1)
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[9]:


# f_0 = 'x**3 + x - 1'
# f_1 = str(f_0)
f = lambda x: np.sin(x)


# In[10]:


i = 1
h = 0.001
# a = terms
func = np.matrix([f(i-2*h),f(i-h),f(i),f(i+h),(f(i+2*h))])/(h**4)
print(func)


# In[11]:


for i in range(5):
    deriv = np.matmul(func,c)/math.factorial(i)
    print(deriv)


# In[ ]:




