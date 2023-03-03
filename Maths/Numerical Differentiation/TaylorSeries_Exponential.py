#!/usr/bin/env python
# coding: utf-8

# In[708]:


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


# In[709]:


# Import modules from another workbook
# %%capture
get_ipython().run_line_magic('run', 'fourthorder_centrediﬀerence.ipynb')


# In[710]:


# Function
n,sol_val,index = centre_difference("fourth-order")
print(n)
print(sol_val)
print(index)


# In[711]:


# Create System Matrix
mat_list = sys_matrix(n)
# Solution Vector
b = np.zeros((1,n))
# Index
b[0][index] = sol_val
# Create Weights
c = np.linalg.pinv(np.matrix(mat_list))*b.T
print(c)


# In[712]:


#Exponential
# Delta
dx = 0.01*6
# Guess
t= 0
# Derivative Coefficient Term
term = 4
# Exp Function
f = lambda x: np.exp(x)

# Function matrix
func = np.matrix([f(t-3*dx),f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx),f(t+3*dx)])/(dx**4)
# func = np.matrix([f(t-2*dx),f(t-dx),f(t),f(t+dx),f(t+2*dx)])/(dx**2)
print(func)

# Derivative coefficient
deriv = (np.matmul(func,c))/6/math.factorial(term)
print(deriv)


# In[713]:


# Differentation
# Setup Differentation Counter
diff_counter = term
# Conduct differentation
deriv_diff = deriv*diff_counter
print(deriv_diff)
# Update differentation counter
diff_counter = diff_counter - 1


# In[ ]:




