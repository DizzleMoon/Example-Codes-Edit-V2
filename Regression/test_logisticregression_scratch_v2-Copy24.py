#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]


# In[3]:


x = [(1,) + row[:2] for row in data]
y = [row[2] for row in data]


# In[4]:


plt.scatter([row[0] for row in data if row[2]],
            [row[1] for row in data if row[2]],
            marker='.', label='paid')
plt.scatter([row[0] for row in data if not row[2]],
            [row[1] for row in data if not row[2]],
            marker='+', label='unpaid')
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.title('Paid and Unpaid Users')
plt.show()


# In[5]:


# Datasets
exp = []
salary = []
binary_cnt = []

for i in range(0,len(data)):
    exp.append(data[i][0])
    salary.append(data[i][1])
    binary_cnt.append(data[i][2])


# In[6]:


# Intialize matrix

# Matrix Size: Degree of Polynomial
poly_deg = 1
mat_size = poly_deg + 1

# Create dummy matrix
a = np.ones((mat_size,mat_size))
b = np.ones((mat_size,1))

y = salary
actual_y = salary

x_ones = np.ones((len(data),1))
x_exp = np.matrix(exp)
x_exp = np.transpose(x_exp)

x3 = np.hstack((x_ones,x_exp))
y = np.matrix(y)

# Least Squares Method
b_0 = np.matmul(x3.T, x3)
# b_0 = x3.T * x3
b_1 = np.linalg.pinv(b_0)
salary = np.matrix(salary)
b_2 = np.matmul(x3.T, salary.T)
b_hat = np.dot(b_1,b_2)
print("b_hat:", b_hat)
# ab0 = b_hat[1:][:,0]
# ab1 = b_hat[1:][:,1]


# In[7]:


xx1 = np.linspace(min(exp), max(exp), len(exp))
xx0 = np.linspace(0, max(salary), len(salary))
zz = np.array(b_hat[0] + b_hat[1]* xx1)


# In[8]:


# exp_paid = []
# sal_paid = []
# [exp_paid.append(row[0]) for row in data if row[2]]
# [sal_paid.append(row[1]) for row in data if row[2]]

# exp_unpaid = []
# sal_unpaid = []
# [exp_unpaid.append(row[0]) for row in data if not row[2]]
# [sal_unpaid.append(row[1]) for row in data if not row[2]]


# In[9]:


exp_paid = []
sal_paid = []
exp_unpaid = []
sal_unpaid = []

for row in data:
    if row[2] == 1:
        exp_paid.append(row[0])
        sal_paid.append(row[1])
    else:
        exp_unpaid.append(row[0])
        sal_unpaid.append(row[1])


# In[10]:


# Intialize matrix

# Matrix Size: Degree of Polynomial
poly_deg = 2
mat_size = poly_deg + 1

# Create dummy matrix
a = np.ones((mat_size,mat_size))
b = np.ones((mat_size,1))

y_paid = sal_paid
actual_y = sal_paid

x_ones = np.ones((len(exp_paid),1))
x_exp_paid = np.matrix(exp_paid)
x_exp_paid = np.transpose(x_exp_paid)

x3 = np.hstack((x_ones,x_exp_paid))
y = np.matrix(y)

# Least Squares Method
b_0 = np.matmul(x3.T, x3)
# b_0 = x3.T * x3
b_1 = np.linalg.inv(b_0)
salary = np.matrix(sal_paid)
b_2 = np.matmul(x3.T, salary.T)
b_hat_paid = np.matmul(b_1,b_2)
print("b_hat:", b_hat_paid)


# In[11]:


# Intialize matrix

# Matrix Size: Degree of Polynomial
poly_deg = 2
mat_size = poly_deg + 1

# Create dummy matrix
a = np.ones((mat_size,mat_size))
b = np.ones((mat_size,1))

y_unpaid = sal_unpaid
# actual_y = sal_paid

x_ones = np.zeros((len(exp_unpaid),1))
x_exp_unpaid = np.matrix(exp_unpaid)
x_exp_unpaid = np.transpose(x_exp_unpaid)

x3 = np.hstack((x_ones,x_exp_unpaid))
y = np.matrix(y)

# Least Squares Method
b_0 = np.matmul(x3.T, x3)
# b_0 = x3.T * x3
b_1 = np.linalg.pinv(b_0)
salary = np.matrix(sal_unpaid)
b_2 = np.matmul(x3.T, salary.T)
b_hat_unpaid = np.matmul(b_1,b_2)
print("b_hat:", b_hat_unpaid)


# In[12]:


xx1 = np.linspace(min(exp), max(exp), len(exp))
xx0 = np.linspace(0, max(salary), len(salary))
zz_paid = np.array(b_hat_paid[0] + b_hat_paid[1] * xx1)
zz_unpaid = np.array(b_hat_unpaid[0] + b_hat_unpaid[1] * xx1)


# In[13]:


# # Plots

# Actual & Predicted values
plot1 = plt.figure(1)
# plt.scatter([row[0] for row in data if row[2]],
#             [row[1] for row in data if row[2]],
#             marker='.', label='paid')
# plt.scatter([row[0] for row in data if not row[2]],
#             [row[1] for row in data if not row[2]],
#             marker='+', label='unpaid')
plt.scatter(exp_paid, sal_paid, marker='.', label='paid')
plt.scatter(exp_unpaid, sal_unpaid, marker='+', label='unpaid')
# plt.plot(xx1.T,zz_paid.T)
# plt.plot(xx1.T,zz_unpaid.T)
plt.plot(xx1.T, (zz_unpaid.T+zz_paid.T + zz.T)/3)
# plt.plot(xx1.T,zz.T)
plt.xlabel('experience')
plt.ylabel('salary')
plt.legend()
plt.title('Paid and Unpaid Users')
# plt.scatter(dist, actual_y)
# plt.plot(x_list, J_tot)
# plt.plot(x_list, J_tot_smooth)
# plt.legend(["Initial Plot", "Predicted Plot", "Predicted Trendline"])
plt.ticklabel_format(useOffset=False)
plt.grid()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




