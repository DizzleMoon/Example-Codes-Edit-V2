#!/usr/bin/env python
# coding: utf-8

# In[32]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
# import pygal
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


# In[33]:


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


# stats.describe(rent)


# In[34]:


# Square Feet
sqft_mean_full = np.mean(sqft)
print(sqft_mean_full)
sqft_mean = np.mean(sqft[:-1])
print(sqft_mean)


# In[35]:


# Rent
rent_mean_full = np.mean(rent)
print(rent_mean_full)
rent_mean = np.mean(rent[:-1])
print(rent_mean)


# In[36]:


resd_1 = []
for i in range(len(rent)):
    resd_1.append(rent[i] - rent_mean_full )
    
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

# Delta 2
delta_pos = []
delta_neg = []
for i in range(0,len(rent)):
    if i < n+1:
        delta_pos.append(resd_1[i])
    else:
        delta_neg.append(resd_1[i])
# delta_pos
d_pos = np.mean(delta_pos)
d_pos = d_pos * np.ones((n+1,1))
d_pos

# if y_f2_mean == 0:
#     y_f2_ind = o
#     y_f2_mean = np.mean(y_f2[0:y_f2_ind])
#     d_pos = y_f2_mean * np.ones((y_f2_ind,1))
#     d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
# d_pos    


# delta_neg
d_neg = np.mean(delta_neg)
d_neg = d_neg * np.ones((o-1,1))
d_neg

delta_final = np.vstack((d_pos,delta_neg))
delta_final


# In[37]:


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


# In[38]:


#y - F_1
y_f1 = []
for i in range(len(rent)):
    y_f1.append(rent[i] - f_1[i]) 
y_f1


# In[39]:


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


# In[40]:


# F2
f_2 = []
for i in range(len(rent)):
    f_2.append(f_1[i] + delta_2[i])
f_2


# In[41]:


# y - F2
y_f2 = []
for i in range(len(rent)):
    y_f2.append(rent[i] - f_2[i]) 
y_f2


# In[42]:


np.median(y_f2)


# In[43]:


# delta T
delta_3 = np.mean(y_f2[0:len(rent)-1]) * np.ones((len(rent)-1,1))
delta_3 = np.vstack((delta_3,y_f2[len(rent)-1]))
print(delta_3)


# In[44]:


# F3
f_3 = []
for i in range(len(rent)):
    f_3.append(f_2[i] + delta_3[i])
f_3


# In[45]:


# y - F3
y_f3 = []
for i in range(len(rent)):
    y_f3.append(rent[i] - f_3[i]) 
y_f3


# In[46]:


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


# In[47]:


find_best_split(sqft,rent)


# In[48]:


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


# In[49]:


sqft


# In[50]:


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


# In[51]:


# y_f1
# y_f1_mean = np.mean(y_f1)

# abc = []
# abc_init = np.square(y_f1[0] - y_f1_mean)
# abc.append(abc_init)
# for i in range(1,5):
#     var_0 = np.sum((y_f1[i]-y_f1_mean)**2)
#     abc.append(var_0/(i+1))
# # abcd = np.sum(abc)/5
# # abcd
# abc

# # m = min(filter(lambda x: x > 0, abc))
# # m = min(i for i in abc if i > 0)
# a = 0
# for i in abc:
#     if i > 0:
#         m = i
#         a = 1
# n = abc.index(m)
# if a == 1:
#     n = 1
# o = len(y_f1) - n
# o

# # Delta 2
# delta_pos = []
# delta_neg = []
# for i in range(1,len(rent)):
#     if i < n+1:
#         delta_pos.append(y_f1[i])
#     else:
#         delta_neg.append(y_f1[i])
# delta_pos
# d_pos = np.mean(delta_pos)
# d_pos = d_pos * np.ones((n+1,1))
# d_pos

# delta_neg
# d_neg = np.mean(delta_neg)
# d_neg = d_neg * np.ones((o,1))
# d_neg

# delta_final = np.vstack((d_pos,d_neg))
# delta_final


# In[52]:


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

# if y_f2_mean == 0:
#     y_f2_ind = o
#     y_f2_mean = np.mean(y_f2[0:y_f2_ind])
#     d_pos = y_f2_mean * np.ones((y_f2_ind,1))
#     d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
# d_pos    

delta_final = np.vstack((d_pos,d_neg))
delta_final


# In[53]:


# y_f2 = np.array(y_f2)
# y_f2_mean = np.mean(y_f2)

# # if y_f2_mean == 0:
# #     y_f2_mean = np.mean(y_f2[0:len(y_f2)-1])

# abc = []
# abc_init = np.square(y_f2[0] - y_f2_mean)
# abc.append(abc_init)
# for i in range(1,5):
#     var_0 = np.sum((y_f2[i]-y_f2_mean)**2)
#     abc.append(var_0/(i+1))
# # abcd = np.sum(abc)/5
# # abcd
# abc


# a_lst = []
# for i in abc:
#     if i > 0:
#         a_lst.append(i)
# #         a = 1
# # n = abc.index(m)
# # if a == 1:
# #     n = n - 1
# # o = len(y_f1) - n

# # m
# # Min
# min_a = min(a_lst)
# # min_a
# n = abc.index(min_a)
# o = len(rent) - n

# # Delta 2
# delta_pos = []
# delta_neg = []
# for i in range(0,len(rent)):
#     if i < n:
#         delta_pos.append(y_f2[i])
#     else:
#         delta_neg.append(y_f2[i])
# delta_pos
# d_pos = np.mean(delta_pos)
# d_pos = d_pos * np.ones((n+1,1))

# delta_neg
# d_neg = np.mean(delta_neg)
# d_neg = d_neg * np.ones((o,1))
# d_neg

# # if y_f2_mean == 0:
# #     y_f2_ind = len(y_f2)-1
# #     y_f2_mean = np.mean(y_f2[0:y_f2_ind])
# #     d_pos = y_f2_mean * np.ones((y_f2_ind,1))
# #     d_neg = delta_neg[3]
    
# delta_neg

# # delta_final = np.vstack((d_pos,d_neg))
# # delta_final
# # delta_neg[len(y_f2)-1]


# In[54]:


y_f2 = np.array(y_f2)
y_f2_mean = np.mean(y_f2)

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
print("O: ", o)

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
d_pos = d_pos * np.ones((n+1,1))

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
delta_final
# delta_neg[len(y_f2)-1]


# In[55]:


# F3
f_3 = []
for i in range(len(rent)):
    f_3.append(f_2[i] + delta_final[i])
f_3


# In[56]:


# y - F3
y_f3 = []
for i in range(len(rent)):
    y_f3.append(rent[i] - f_3[i]) 
y_f3


# In[57]:


# y_f3 = np.array(y_f3)
# y_f3_mean = np.mean(y_f3)

# # if y_f2_mean == 0:
# #     y_f2_mean = np.mean(y_f2[0:len(y_f2)-1])

# abc = []
# abc_init = np.square(y_f3[0] - y_f3_mean)
# abc.append(abc_init)
# for i in range(1,5):
#     var_0 = np.sum((y_f3[i]-y_f3_mean)**2)
#     abc.append(var_0/(i+1))
# # abcd = np.sum(abc)/5
# # abcd
# abc


# a_lst = []
# for i in abc:
#     if i > 0:
#         a_lst.append(i)
# #         a = 1
# # n = abc.index(m)
# # if a == 1:
# #     n = n - 1
# # o = len(y_f1) - n

# # m
# # Min
# min_a = min(a_lst)
# # min_a
# n = abc.index(min_a)
# o = len(rent) - n

# # # Delta 2
# # delta_pos = []
# # delta_neg = []
# # for i in range(0,len(rent)):
# #     if i < n:
# #         delta_pos.append(y_f3[i])
# #     else:
# #         delta_neg.append(y_f3[i])
# # delta_pos
# # d_pos = np.mean(delta_pos)
# # d_pos = d_pos * np.ones((n+1,1))

# # delta_neg
# # d_neg = np.mean(delta_neg)
# # d_neg = d_neg * np.ones((o,1))
# # d_neg

# if y_f3_mean == 0:
#     y_f3_ind = n 
#     y_f3_mean = np.mean(y_f3[0:y_f3_ind])
#     d_pos = y_f3_mean * np.ones((y_f3_ind,1))
#     d_neg = delta_neg[n-o] * np.ones((len(rent)-n,1))
# d_pos    
# # # delta_neg[y_f2_ind-1]
# d_neg
# # # n
# # delta_final = np.vstack((d_pos,d_neg))
# # delta_final
# # # delta_neg[len(y_f2)-1]


# In[58]:


y_f2 = y_f3
y_f2 = np.array(y_f2)
y_f2_mean = np.mean(y_f2)

# if y_f2_mean == 0:
#     y_f2_mean = np.mean(y_f2[0:len(y_f2)-1])

abc = []
abc_init = np.square(y_f2[0] - y_f2_mean)
abc.append(abc_init)
for i in range(1,5):
    var_0 = np.sum((y_f2[i]-y_f2_mean)**2)
    abc.append(var_0/(i+1))
# abcd = np.sum(abc)/5
abc = abc[0].tolist() + abc[1:5]



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
print("O: ", o)

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
d_pos = d_pos * np.ones((n+1,1))

delta_neg
d_neg = np.mean(delta_neg)
d_neg = d_neg * np.ones((o,1))

# d_pos
# d_neg

if y_f2_mean == 0:
    y_f2_ind = o
#     y_f2_mean = np.mean(y_f2[0:y_f2_ind])
    y_f2_mean = np.mean(np.array(a_lst))
    d_pos = y_f2_mean * np.ones((y_f2_ind,1))
    d_neg = delta_neg[o-n] * np.ones((len(rent)-o,1))
d_pos    
# # delta_neg[y_f2_ind-1]
d_neg
# # n
delta_final = np.vstack((d_pos,d_neg))
delta_final
# delta_neg[len(y_f2)-1]
# y_f2_mean


# In[ ]:




