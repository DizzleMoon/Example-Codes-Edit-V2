#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import random
import sys
import numpy as np
from stats import mean, de_mean, standard_deviation, correlation
from gradient_descent import minimize_stochastic
import matplotlib.pyplot as plt


# In[2]:


data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")
data[:5,::] #dataset loaded demonstration


# In[3]:


# Break datasets into X and Y.
X_0 = np.array(data[::,0:2])
Y = data[::,-1:]


# In[4]:


# Setup bias array
X_ones = np.ones(len(X_0))
# Concentate arrays
# X = [X_ones,X_0]
X1 = np.vstack((X_ones.T,X_0.T))
# X1 = [[X_ones],[X_0]]
X1.T


# In[5]:


Theta = np.linalg.pinv(X1.dot(X1.T)).dot(X1).dot(Y)


# In[6]:


# Feature Scaling
# Mean
mean_size = np.mean(X1[1])
mean_bedroom = np.mean(X1[2])
# Standard Deviation
std_size = np.std(X1[1])
std_bedroom = np.std(X1[2])
# Scaling
X1[1] = (X1[1] - mean_size)/std_size
X1[2] = (X1[2] - mean_bedroom)/std_bedroom


# In[7]:


#define function to find cost
def cost_function(X, y, theta):
    """
    cost_function(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    """
    ## number of training examples
    m = len(y) 
    
    ## Calculate the cost with the given parameters
    J = np.sum((X.dot(theta)-y)**2)/2/m
    
    return J


# In[8]:


def gradientDescent(X,y,theta,alpha,num_iters):
    """
    Take in numpy array X, y and theta and update theta by taking   num_iters gradient steps
    with learning rate of alpha
    
    return theta and the list of the cost of theta during each  iteration
    """
    
    m=len(y)
    J_history=[]
    
    for i in range(num_iters):
        predictions = X.dot(theta)
#         error = np.dot(X.transpose(),(predictions -y))
        error = X.T.dot(predictions-y)
        descent=alpha * 1/m * error
        theta = theta - descent
        J_history.append(cost_function(X, y, theta))
    
    return theta,J_history


# In[9]:


alpha = 0.3
iterations = 100
t,J = gradientDescent(X1.T,Y,Theta,alpha,iterations)


# In[10]:


# predict the price of a house with 1650 square feet and 3 bedrooms
# add bias unit 1.0
X_predict = np.array([1.0,1650.0,3]) 
#feature scaling the data first
X_predict[1] = (X_predict[1] - mean_size)/ (std_size) 
X_predict[2] = (X_predict[2]- mean_bedroom)/ (std_bedroom)
hypothesis = X_predict.dot(t)
print("Cost of house with 1650 sq ft and 3 bedroom is ",hypothesis)

