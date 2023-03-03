#!/usr/bin/env python
# coding: utf-8

# In[49]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
import sys
import os
import string
import urllib.request
import requests
import curl
import pycurl
import webbrowser
import numpy as np
import math
import pandas as pd
# from IPython import qt
from matplotlib.pyplot import figure
from py.xml import raw
from requests.api import get
from matplotlib import pyplot as plt
# from scratch.working_with_data import rescale
# from scratch.multiple_regression import least_squares_fit, predict
# from scratch.gradient_descent import gradient_step

# from stats import mean, median, de_mean, standard_deviation, correlation
# from gradient_descent import minimize_stochastic, maximize_stochastic, maximize_batch
# from vector import dot, vector_add
# from normal import normal_cdf
# from matrix import make_matrix, get_column, shape, matrix_multiply
# from logistic_regression import *

import math
import os
import random
import sys
from functools import partial, reduce

from scipy.optimize import fmin_tnc

import tqdm

from typing import*

from collections import*
from scipy import*
from sklearn.metrics import*


# In[50]:


# Room Size
sqft = np.array([750,800,850,900,950])
# Rent
rent = np.array([1160,1200,1280,1450,2000])
# Average Rent
f_0 = []
for _ in range(0,len(rent)):
    f_0.append(np.mean(rent))
# Residuals
resd = rent-f_0
resd

# 
# stats.describe(rent)


# In[51]:


# Square Feet
sqft_mean_full = np.mean(sqft)
print(sqft_mean_full)
sqft_mean = np.mean(sqft[:-1])
print(sqft_mean)


# In[52]:


# Rent
rent_mean_full = np.mean(rent)
print(rent_mean_full)
rent_mean = np.mean(rent[:-1])
print(rent_mean)


# In[53]:


resd_1 = []
for i in range(len(rent)):
    resd_1.append(rent[i] - rent_mean_full )
    
y_f2_mean = []
resd_1

resd_1
resd_1_mean = np.mean(resd_1)

abc = []
abc_init = np.square(resd_1[0] - resd_1_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((resd_1[i]-resd_1_mean)**2)
    abc.append(var_0/(i))
# abcd = np.sum(abc)/5
# abcd
abc

# m = min(filter(lambda x: x > 0, abc))
# m = min(i for i in abc if i > 0)
# n = abc.index(m)
# n
# o = len(rent) - n

# a = 0
a_lst = []
for i in abc:
    if i > 0:
        a_lst.append(i)
#         a = 1
# n = abc.index(m)
# if a == 1:
#     n = n - 1
# o = len(y_f1) - n

# m
# Min
min_a = min(a_lst)
# min_a
n = abc.index(min_a)
o = len(rent) - n

print(n)
print(o)

if n > o:
    n = n + 1

# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if i < n:
        delta_pos.append(resd_1[i])
    else:
        delta_neg.append(resd_1[i])
# delta_pos
d_pos = np.mean(delta_pos)
d_pos = d_pos * np.ones((n,1))
d_pos

if y_f2_mean == 0:
    y_f2_ind = o
    y_f2_mean = np.mean(y_f2[0:y_f2_ind])
    d_pos = y_f2_mean * np.ones((y_f2_ind,1))
    d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
d_pos    


# delta_neg
d_neg = np.mean(delta_neg)
d_neg = d_neg * np.ones((o-1,1))
d_neg

delta_final_1 = np.vstack((d_pos,delta_neg))
print(delta_final_1)

resd_1


# In[54]:


# delta T
delta_1 = np.mean(resd[0:len(rent)-1]) * np.ones((len(rent)-1,1))
print(delta_1)
# resd[len(rent)-1]
# delta_1 = np.vstack((delta_1,resd[len(rent)-1]))
# f_1 = np.mean(f_0[0:len(rent)-1])* np.ones((len(rent)-1,1)) + delta_1
# f_1
# f_0_1 = np.vstack(((np.mean(f_0[0:len(rent)])* np.ones((len(rent)-1,1))),rent[len(rent)-1])) 
f_0_1 = np.mean(f_0[0:len(rent)])* np.ones((len(rent)-1,1))
f_0_1 = f_0_1 + delta_1
f_1 = np.vstack((f_0_1,rent[len(rent)-1])) 
f_1


# In[55]:


#y - F_1
y_f1 = []
for i in range(len(rent)):
    y_f1.append(rent[i] - f_1[i]) 
y_f1


# In[56]:


# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if y_f1[i] >= 0:
        delta_pos.append(y_f1[i])
    else:
        delta_neg.append(y_f1[i])
delta_pos

delta_2 = []
for i in range(0,len(rent)):
    if y_f1[i] >= 0:
        delta_2.append(np.mean(delta_pos))
    else:
        delta_2.append(np.mean(delta_neg))
delta_2


# In[57]:


# F2
f_2 = []
for i in range(len(rent)):
    f_2.append(f_1[i] + delta_2[i])
f_2


# In[58]:


# y - F2
y_f2 = []
for i in range(len(rent)):
    y_f2.append(rent[i] - f_2[i]) 
y_f2


# In[59]:


np.median(y_f2)


# In[60]:


# delta T
delta_3 = np.mean(y_f2[0:len(rent)-1]) * np.ones((len(rent)-1,1))
delta_3 = np.vstack((delta_3,y_f2[len(rent)-1]))
print(delta_3)


# In[61]:


# F3
f_3 = []
for i in range(len(rent)):
    f_3.append(f_2[i] + delta_3[i])
f_3


# In[62]:


# y - F3
y_f3 = []
for i in range(len(rent)):
    y_f3.append(rent[i] - f_3[i]) 
y_f3


# In[63]:


def find_best_split(x,y):
    best_loss = np.inf
    best_split = -1
    print(f"find_best_split in x={list(x)}")
    for v in x[1:]: # try all possible x values
        lefty = y[x<v]
        righty = y[x>=v]
        nl = len(lefty)
        nr = len(righty)
        if nl==0 or nr==0:
            continue
        # variance is same as MSE here
        # weight by proportion on left and right, get avg as loss
        loss = (np.var(lefty)*nl + np.var(righty)*nr)/2
        print(f"{lefty} | {righty}    candidate split x ={v:4d} loss {loss:8.1f}")
        if loss < best_loss:
            best_loss = loss
            best_split = v
    return float(best_loss), best_split


# In[64]:


find_best_split(sqft,rent)


# In[65]:


x = sqft
y = rent
best_loss = np.inf
best_split = -1
print(f"find_best_split in x={list(x)}")
for v in x[1:]:
    left_y = y[x<v]
    right_y = y[x >= v]

    nl = len(left_y)
    nr = len(right_y)
    
    if nl == 0 or nr == 0:
        continue
    loss = (np.var(left_y)*nl + np.var(right_y)*nr)/2
    print(f"{left_y} | {right_y}    candidate split x ={v:4d} loss {loss:8.1f}")
    if loss < best_loss:
        best_loss = loss
        best_split = v
    
    
print(left_y)
print(right_y)
print(loss)
print(best_loss)
print(best_split)


# In[66]:


sqft


# In[67]:


sqft_mean = np.mean(sqft)
summ = []

# for q in range(1,len(sqft)):
#     sum1 = np.sum(sqft[:q])
#     summ.append(sum1)
# #     for p in summ:
# #         print(p)
# summ
abc = []
abc_init = np.square(sqft[0] - sqft_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((sqft[i]-sqft_mean)**2)
    abc.append(var_0/(i+1))
# abcd = np.sum(abc)/5
# abcd
abc


# In[68]:


y_f1
y_f1_mean = np.mean(y_f1)

abc = []
abc_init = np.square(y_f1[0] - y_f1_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((y_f1[i]-y_f1_mean)**2)
    abc.append(var_0/(i+1))
# abcd = np.sum(abc)/5
# abcd
abc


a_lst = []
for i in abc:
    if i > 0:
        a_lst.append(i)
#         a = 1
# n = abc.index(m)
# if a == 1:
#     n = n - 1
# o = len(y_f1) - n

# m
# Min
min_a = min(a_lst)
# min_a
n = abc.index(min_a)
o = len(rent) - n

print("n: ",n)
print("o: ",o)

if n > o:
    n = n + 1

# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if i < n:
        delta_pos.append(y_f1[i])
    else:
        delta_neg.append(y_f1[i])
delta_pos
d_pos = np.mean(delta_pos)
d_pos = d_pos * np.ones((n,1))
d_pos

delta_neg
d_neg = np.mean(delta_neg)
d_neg = d_neg * np.ones((o,1))
d_neg

if y_f2_mean == 0:
    y_f2_ind = o
    y_f2_mean = np.mean(y_f2[0:y_f2_ind])
    d_pos = y_f2_mean * np.ones((y_f2_ind,1))
    d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
d_pos    

delta_final = np.vstack((d_pos,d_neg))
delta_final


# In[69]:


y_f2 = np.array(y_f2)
y_f2_mean = np.mean(y_f2)
print(y_f2_mean)

# if y_f2_mean == 0:
#     y_f2_mean = np.mean(y_f2[0:len(y_f2)-1])

abc = []
abc_init = np.square(y_f2[0] - y_f2_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((y_f2[i]-y_f2_mean)**2)
    abc.append(var_0/(i+1))
# abcd = np.sum(abc)/5
# abcd
abc


a_lst = []
for i in abc:
    if i > 0:
        a_lst.append(i)
#         a = 1
# n = abc.index(m)
# if a == 1:
#     n = n - 1
# o = len(y_f1) - n

# m
# Min
min_a = min(a_lst)
# min_a
n = abc.index(min_a)
o = len(rent) - n
print("N: ", n)
print("O: ",o)

if n > o:
    n = n + 1

# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if i < n:
        delta_pos.append(y_f2[i])
    else:
        delta_neg.append(y_f2[i])
delta_pos
d_pos = np.mean(delta_pos)
d_pos = d_pos * np.ones((n,1))

delta_neg
d_neg = np.mean(delta_neg)
d_neg = d_neg * np.ones((o,1))
d_neg

if y_f2_mean == 0:
    y_f2_ind = o
    y_f2_mean = np.mean(y_f2[0:y_f2_ind])
    d_pos = y_f2_mean * np.ones((y_f2_ind,1))
    d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
d_pos    
# delta_neg[y_f2_ind-1]
# d_neg
# n
delta_final = np.vstack((d_pos,d_neg))
print(delta_final)
# delta_neg[len(y_f2)-1]
# delta_neg
print(delta_neg)
print(delta_pos)
print(y_f2_mean)
print(y_f2_ind)
print(y_f2[0:o])


# In[70]:


# F3
f_3 = []
for i in range(len(rent)):
    f_3.append(f_2[i] + delta_final[i])
f_3


# In[71]:


# y - F3
y_f3 = []
for i in range(len(rent)):
    y_f3.append(rent[i] - f_3[i]) 
y_f3


# In[72]:


# y - F3
y_f3 = []
for i in range(len(rent)):
    y_f3.append(rent[i] - f_3[i]) 
y_f3

# y_f3_ind = y_f3.index(0)
# y_f3_ind


# In[73]:


y_f3 = np.array(y_f3)
y_f3_mean = np.mean(y_f3)

# if y_f2_mean == 0:
#     y_f2_mean = np.mean(y_f2[0:len(y_f2)-1])

abc = []
abc_init = np.square(y_f3[0] - y_f3_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((y_f3[i]-y_f3_mean)**2)
    abc.append(var_0/(i+1))
# abcd = np.sum(abc)/5
# abcd
abc = abc[0].tolist() + abc[1:5]
abc


a_lst = []
for i in abc:
    if i > 0:
        a_lst.append(i)
#         a = 1
# n = abc.index(m)
# if a == 1:
#     n = n - 1
# o = len(y_f1) - n
a_lst

# m
# Min
min_a = min(a_lst)
# min_a
n = abc.index(min_a)
o = len(rent) - n
print("N: ", n)
print("O: ", o)

if n > o:
    n = n+1

# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if i < n:
        delta_pos.append(y_f3[i])
    else:
        delta_neg.append(y_f3[i])
delta_pos
# d_pos = np.mean(delta_pos)
# d_pos = d_pos * np.ones((n+1,1))
# d_pos
delta_neg
# d_neg = np.mean(delta_neg)
# d_neg = d_neg * np.ones((o,1))
# d_neg

if y_f3_mean == 0:
    y_f3_ind = o
    print(y_f3_ind)
    if y_f3_ind >= 0:
        y_f3_mean = np.mean(y_f3[0:y_f3_ind-1])
        d_pos = y_f3_mean * np.ones((y_f3_ind,1))
    #     d_neg = y_f3_mean
        d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
    else:
        y_f3_mean = np.mean(y_f3[0:y_f3_ind])
        d_pos = y_f3_mean * np.ones((y_f3_ind,1))
    #     d_neg = y_f3_mean
        d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
# d_pos    
# # delta_neg[y_f2_ind-1]
# d_neg
# # n
delta_final = np.vstack((d_pos,d_neg))
# # delta_neg[len(y_f2)-1]
print(delta_neg)
print(delta_pos)
print(d_pos)
print(d_neg)
print(y_f3_mean)
print(delta_final)


# In[74]:


if y_f3_mean == 0:
    y_f3_ind = o
    y_f3_mean = np.mean(y_f3[0:y_f3_ind])
    d_pos = y_f3_mean * np.ones((y_f3_ind,1))
    d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
d_pos 
d_neg

delta_final = np.vstack((d_pos,d_neg))
print(delta_final)


# In[75]:


# F4
f_4 = []
for i in range(len(rent)):
    f_4.append(f_3[i] + delta_final[i])
f_4


# In[76]:


# y - F4
y_f4 = []
for i in range(len(rent)):
    y_f4.append(rent[i] - f_4[i]) 
y_f4

# y_f3_ind = y_f3.index(0)
# y_f3_ind


# In[77]:


rms = mean_absolute_error(rent,f_4)
rms


# In[78]:


MSE = np.square(np.subtract(rent,f_2)).mean() 
MSE


# In[79]:


# Plot 1
ff = []
for i in range(len(rent)):
    ff.append(f_0[i] + delta_final_1[i] + delta_2[i] + delta_3[i])
# ff = np.multiply(0.5,ff)


# In[80]:


plt.scatter(sqft,rent)
plt.plot(sqft,ff, marker='*', color='red')
plt.grid()
plt.show()


# In[81]:


ff


# In[82]:


sqft[len(ff)//2]
sqft
ff[-2]


# In[83]:


# Create stepped plots
sqft_2 = []
rent_2 = []
for i in range(1,len(ff)):
    sqft_2.append(sqft[i-1])
    rent_2.append(ff[i])
    if abs(ff[i]-ff[i-1]) != 0:
#         print((sqft[i] + sqft[i-1])/2)
        sqft_2.append((sqft[i] + sqft[i-1])/2)
        sqft_2.append((sqft[i] + sqft[i-1])/2)
        rent_2.append(ff[1])   
        rent_2.append(ff[i-1])
    if i == len(ff)-1:
        print(i)
        sqft_2.append(sqft[len(rent)-1])
        rent_2.append(rent[len(rent)-1])  
        
rent_2 = np.sort(rent_2)      
print(rent_2)
print(sqft_2)


# In[84]:


for j in range(1,len(rent_2)):
    if sqft_2[j] - sqft_2[j-1] == 0:
        rent_2[j] = rent_2[j+1]
rent_2


# In[85]:


rent


# In[86]:


plt.scatter(sqft,rent)
plt.scatter(sqft_2,rent_2, marker='+', color='darkviolet')
plt.plot(sqft_2,rent_2, color='darkorange')
# plt.plot(sqft,ff, marker='*', color='red')
plt.grid()
plt.show()


# In[87]:


ff


# In[ ]:




