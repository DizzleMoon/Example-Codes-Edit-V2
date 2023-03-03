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

from scipy.optimize import fmin_tnc

import tqdm

from typing import*

from collections import*


# In[2]:


# Imports dataset
df = pd.read_csv('Flight Delays Data.csv')
df.head()


# In[3]:


# ## Create a vector of 0 of length equal to the number of rows
# temp = [0] * df.shape[0]
# temp = np.zeros((df.shape[0],1))
# cnt_out = 0
# ## test each outlier condition and mark with a 1 as required
# for i, x in enumerate(df['ArrDelay']):
#         if (x > 100): 
#             temp[i] = 1
#             cnt_out += 1
    

# # # for i, x in enumerate(df['DepDelay']):
# # #     if (x > 100): temp[i] = 1 
# # # for i, x in enumerate(df['Carrier']):
# # #     if (x > 40): temp[i] = 1      
# df['Outlier'] = temp # append a column to the data frame
# df


# In[4]:



# # for col in plot_cols: # loop over the columns
# fig = plt.figure(figsize=(6, 6))
# ax = fig.gca()
# ## Loop over the zip of the four vectors an subset the data and
# ## create the plot using the aesthetics provided
# # for o, f, c, m in zip(outlier, fuel, color, marker):
# for o in df['Outlier']:
#     temp = df['Outlier'].loc[(df['Outlier'] == o)] 
#     print(temp)
# #     if temp > 0:                    
# #         temp.plot(kind = 'scatter', x = df['ArrDelay'][o], y = df['DepDelay'][o] , 
# #                    ax = ax)                                 
# # ax.set_title('Scatter plot of lnprice vs. ' + col)


# In[5]:


qnt = df['ArrDelay'].quantile([.25,.75])
print(qnt)

# IQR
# # First quartile (Q1) 
# Q1 = np.median(df['ArrDelay']) 
  
# # Third quartile (Q3) 
# Q3 = np.median(df['ArrDelay']) 

Q1 = df['ArrDelay'].quantile(0.25)
Q3 = df['ArrDelay'].quantile(0.75)
IQR = Q3 - Q1


upper = qnt.iloc[1] + (1.5*IQR)
lower = qnt.iloc[0] - (1.5*IQR)

print(Q1,Q3)
print(qnt.iloc[0],qnt.iloc[1])

df['ArrDelay'].shape[0]


# In[6]:


df[df['ArrDelay'] > upper] = None
df[df['ArrDelay'] < lower] = None
# df.dropna()
new_data = df.dropna(axis = 0, how ='any')
# len(df)
# df['ArrDelay'].shape[0]

# dd = 2435411
# print(dd)
new_data.shape[0]
new_data


# In[ ]:





# In[ ]:




