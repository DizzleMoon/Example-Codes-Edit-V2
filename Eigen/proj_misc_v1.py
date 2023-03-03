#!/usr/bin/env python
# coding: utf-8

# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[10]:


sig_a = np.array([[2,0.8],[0.8,0.6]])
print(sig_a)

vec_sig = np.array([-1,1])
print(vec_sig)
vec_sig_2 = np.array([-1.2,-0.2])
print(vec_sig_2)


vec = sig_a.dot(vec_sig).dot(vec_sig_2)
print(vec)

vec_2 = sig_a.dot(vec)
print(np.gradient(vec_2))

# vec_len = vec_sig.dot(vec)/np.linalg.norm(vec)
# print(vec_len)

# vec_len = (vec[0])/np.linalg.norm(vec)
# print(vec_len)

# vec_full_len = np.linalg.norm(vec_len)
# print(vec_full_len)



# print(np.dot(vec[0].vec[1]))

print(np.gradient(sig_a))


# In[ ]:





# In[11]:


sig_a = np.array([[2,0.8],[0.8,0.6]])
print(sig_a)

vec_sig_2 = sig_a
veg_sig_2 = np.array([-2.5,-1.0])

vec_fin = sig_a.dot(sig_a)
print(vec_fin)
sig_a/np.linalg.norm(sig_a)


# In[12]:


x = np.array([2.,0.8])
y = np.array([0.8,0.6])

m,b = np.polyfit(x,y,1)
print(m)
print(b)
# print(c)


# In[13]:


sig_a_grad = np.gradient([[2,0.8],[0.8,0.6]])
sig_a_grad


# In[14]:


from statistics import mean
import numpy as np

# xs = np.array([1,2,3,4,5], dtype=np.float64)
# ys = np.array([5,4,6,5,6], dtype=np.float64)

xs = np.array([2.,0.8])
ys = np.array([0.8,0.6])

def best_fit_slope(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(ys**2)))
    return m

m = best_fit_slope(xs,ys)
print(m)

