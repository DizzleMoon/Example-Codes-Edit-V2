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
from matplotlib.pyplot import figure


# In[2]:


# Sample size
Ts = 0.001

# Number of samples
N = 1000

# Sample range
sample_range = np.linspace(0,Ts*N,N)


# In[3]:


# Initializze state matrices
# X = np.array([[0.0], [0.0], [0.1], [0.1]])
X = np.array([[100],[0]])
# P0 = np.diag((0.01, 0.01, 0.01, 0.01))
# P = np.diag((0.01,0.01,0.01,0.01))
P = np.diag((10,0.01))
# A = np.array([[1, 0, Ts , 0], [0, 1, 0, Ts], [0, 0, 1, 0], [0, 0, 0, 1]])
# Q = np.eye(X.shape[0])
Q = np.zeros((2,2))
# B = np.eye(X.shape[0])
# U = np.zeros((X.shape[0],1)) 

# Measurement matrices
Y = np.array([[X[0,0] + abs(np.random.randn(1)[0])], [X[1,0] + abs(np.random.randn(1)[0])]])
H = np.array([[1, 0,], [0, 1]])
# R = np.diag((2,Y.shape[0])) * 
R1 = 0
R0 = np.sqrt(R1)
R = np.array([[2,0],[0,1]])


# In[4]:


# # System matrix - state
A = np.array([[1, Ts], [0,1]])

# # System matrix - input
B = np.array([[-0.5*(Ts**2)], [-Ts]])
# # G = [[-0.5*(Ts**2)], [-Ts]]

# # Input vector
U = 9.80665

# # Observation matrix
# H = np.array([1,0])

# # Sigma - Standard Deviations
# Q = np.array([[0,0],[0,0]])


# In[5]:


x_t = np.zeros((2,N))
x_t[:,0] = [100,0]
for k in range(1,N):
    x_t[:,k] = np.dot(A,x_t[:,k-1]) + np.dot(B.T,U)
R1 = 0
# Measurement noise
v = np.sqrt(R1) + np.random.randn(N)
# Noisy measurement
z = np.matmul(H,x_t) + v

# z
plt.plot(sample_range,z[0])


# In[6]:


# Apply Kalman Filter
for i in range(0,N):
    
    # Prediction
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A,np.dot(P, A.T)) + Q
    plt.plot(X[0,0], X[1,0], 'o', color='red')

    # Update
    nu = np.dot(H,X)
    S = np.dot(H, np.dot(P,H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    X = X + np.dot(K, Y-nu)
    P = P - np.dot(K, np.dot(S,K.T))
    plt.plot(X[0,0], X[1,0], 'o', color='black')
    
    Y = np.array([[X[0,0] + abs(R0 * np.random.randn(1)[0])],[X[1, 0] + abs(R0 * np.random.randn(1)[0])]]) 
    plt.plot(Y[0,0], Y[1,0], color='green')
    
# Y = np.array([[X[0,0] + abs(0.1 * np.random.randn(1)[0])],[X[1, 0] + abs(0.1 * np.random.randn(1)[0])]]) 
# plt.plot(Y[0,:], Y[1,:], color='blue')

# X


# In[7]:


# Apply Kalman Filter
# Prediction
pred_x = []
pred_y = []
# Update
update_x = []
update_y = []
# Measurement
measure_x = []
measure_y = []

X = np.array([[105],[0]])
R = np.array([[4,0],[0,1]])

for i in range(0,N):
    
    # Prediction
    X = np.dot(A,X) + np.dot(B,U)
    P = np.dot(A,np.dot(P, A.T)) + Q
    pred_x.append(X[0,0])
    pred_y.append(X[1,0]) 
    plt.plot(X[0,0], X[1,0], 'o', color='red')
    
    # Update
    nu = np.dot(H,X)
    S = np.dot(H, np.dot(P,H.T)) + R
    K = np.dot(P, np.dot(H.T, np.linalg.pinv(S)))
    X = X + np.dot(K, Y-nu)
    P = P - np.dot(K, np.dot(S,K.T))
    update_x.append(X[0,0])
    update_y.append(X[1,0]) 
    plt.plot(X[0,0], X[1,0], 'o', color='black')
    
    Y = np.array([[X[0,0] + abs(R0  * np.random.randn(N)[0])],[X[1, 0] + abs(R0  * np.random.randn(N)[0])]]) 
    measure_x.append(Y[0,0])
    measure_y.append(Y[1,0]) 
    plt.plot(Y[0,:], Y[1,:], color='blue')
# 
Y


# In[8]:


# Plot

plot1 = plt.figure(1)
plt.plot(pred_x,pred_y, color='green')
# plt.scatter(pred_x,pred_y, color='red')
plt.plot(update_x,update_y)
# plt.scatter(update_x,update_y, color='blue')
plt.plot(measure_x,measure_y, color='blue')
plt.scatter(measure_x,measure_y, color = 'orange')
plt.grid()


# In[9]:


plot2 = plt.figure(2)
# plt.plot(pred_x,pred_y, color='green')
# plt.scatter(pred_x,pred_y, color='red')
plt.plot(sample_range,measure_y,sample_range,update_y)
# plt.scatter(update_x,update_y, color='blue')
# plt.plot(measure_x,measure_y, color='blue')
# plt.scatter(measure_x,measure_y, color = 'orange')
plt.grid()


# In[10]:


plot3 = plt.figure(3)
# plt.plot(pred_x,pred_y, color='green')
plt.scatter(sample_range,pred_x, color='red')
plt.plot(sample_range,z[0],sample_range,measure_x,sample_range,update_x)
# plt.scatter(update_x,update_y, color='blue')
# plt.plot(measure_x,measure_y, color='blue')
# plt.scatter(sample_range,z[0],sample_range,measure_x)
plt.grid()


# In[11]:


plt.figure(4)
plt.plot(sample_range, np.array(pred_y) - np.array(update_y))


# In[ ]:




