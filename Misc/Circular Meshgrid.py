#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import List, Dict, Iterable, Tuple, Callable
from matplotlib import pyplot as plt
from collections import Counter
import pygal
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


# In[8]:


M = 10
N = 100
R1 = 0
R2 = 1
nR = np.linspace(R1, R2, M)
nT = np.linspace(0, 2*np.pi,N)
R,T = np.meshgrid(nR, nT)


# In[71]:


#%%
# Convert grid to cartesian coordinates
X = R * np.cos(T)
Y = R * np.sin(T)
Z = np.ones((len(T),1)) 
row, col = X.shape


# In[72]:


plot1 = plt.figure(1)
ax = plt.axes(projection = '3d')

# for i in range(0,col):
#     plt.plot(X[:,i], Y[:,i])
# for j in range(0,row):
#     plt.plot(X[j,:], Y[j,:])

ax.plot_wireframe(X, Y, Z, cmap = 'viridis', edgecolor = 'green')
# plt.axes(projection = '3d')
# plt.plot(X,Y)
# ax.axis('square')
# plt.axis('square')
plt.grid()
# plt.show()




# In[67]:


A = -1
B = 5.39260384e-33
C = 0


# In[68]:


zz_mesh = A*R + B*T + C
# zz_0 = A*nR + B*nT + C


# In[69]:


plot2 = plt.figure(2)
ax = plt.axes(projection = '3d')

# for i in range(0,col):
# #     ax.plot(X[:,i], Y[:,i])
#     ax.plot_surface(X[:,i], Y[:,i,], zz_mesh, cmap = 'viridis', edgecolor = 'green')
# for j in range(0,row):
# #     ax.plot(X[j,:], Y[j,:])
#     ax.plot_surface(X[j,:], Y[j,:,], zz_mesh, cmap = 'viridis', edgecolor = 'green')

ax.plot_surface(X, Y, zz_mesh, cmap = 'viridis', edgecolor = 'green')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')


# In[70]:


# Array Size

