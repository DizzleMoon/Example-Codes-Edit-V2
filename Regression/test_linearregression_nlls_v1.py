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


def vector_subtract(v, w):
    """subtracts corresponding elements"""
    return [v_i - w_i for v_i, w_i in zip(v, w)]

def scalar_multiply(c, v):
    """c is a number, v is a vector"""
    return [c * v_i for v_i in v]

def in_random_order(data):
#"""generator	that	returns	the	elements	of	data	in	random	order"""
    indexes = [i for i,_ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]

def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    data = list(zip(x, y))
    
    # initial guess
    theta = theta_0
    
    # initial step size
    alpha = alpha_0
    
    # the minimum so far
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9

        # and take a gradient step for each of the data points
        for x_i, y_i in in_random_order(data):
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta

def minimize_stochastic_v0(target_fn, gradient_fn, x, y, theta_0, alpha_0):
    data = list(zip(x, y))
    
    # initial guess
    theta = theta_0
    
    # initial step size
    alpha = alpha_0
    
    # the minimum so far
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
    
    
    # if we ever go 100 iterations with no improvement, stop
#     while iterations_with_no_improvement < 100:
    for _ in range(1000):
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
#             iterations_with_no_improvement = 0
            alpha /= 0.9
        else:
            # otherwise we're not improving, so try shrinking the step size
#             iterations_with_no_improvement += 1
            alpha *= 0.9

        cost_lst = []
        # and take a gradient step for each of the data points
        for x_i, y_i in data:
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))            
#             cost_val = cost(x_i,y_i,theta)
        cost = cost_function(x, y, theta)
        cost_lst.append(cost)
#         print('Cost Val', cost)
            
        
    return min_theta


# In[3]:


data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")
data[:5,::] #dataset loaded demonstration


# In[4]:


# Break datasets into X and Y.
X_0 = np.array(data[::,0:2])
X_1 = np.array(data[::,0])
X_2 = np.array(data[::,1])
Y = data[::,-1:]
X_10 = np.sort(X_1)
X_10 = X_10[::-1]
X_10 = X_10.tolist()
# X_10
# Y


# In[5]:


# Setup bias array
X_ones = np.ones(len(X_0))
# Concentate arrays
# X = [X_ones,X_0]
X1 = np.vstack((X_ones.T,X_0.T))
# X1 = [[X_ones],[X_0]]

# X1.T


# In[6]:


Theta = np.linalg.pinv(X1.dot(X1.T)).dot(X1).dot(Y)
Theta


# In[7]:


# Feature Scaling
# Mean
mean_size = np.mean(X1[1],axis=0)
mean_bedroom = np.mean(X1[2],axis=0)
# Standard Deviation
std_size = np.std(X1[1],axis=0)
std_bedroom = np.std(X1[2],axis=0)
# Scaling
X1[1] = (X1[1] - mean_size)/std_size
X1[2] = (X1[2] - mean_bedroom)/std_bedroom
X11 = X1
X11[1]



Y1 = (Y - np.mean(Y,axis=0))/np.std(Y,axis=0)
# print(Y1)


# In[8]:


x1 = 2.5
x2 = 0.5
x3 = 0.01

# Tolerance
tol = 1e-1
# Iterations
iter = 0
iterations = []
# Coefficients
coeff = []

leng = len(Y)
y = Y


# In[9]:


while tol > 1e-9:
    # Number of iterations
    iter = iter + 1
    iterations.append(iter)
    
    #%%
    # Setup Equations

    # Intialize Array
    F = []
    
    # Vector valued function
    for i in range(0,leng):
        # f = (x1 * math.exp(x2 * t[i])) - y[i]
        f = (x1 * X1[1][i] + x2 * X1[2][i] + x3) - y[i]
        F.append(f)
    # Convert to Array
    F = np.array(F)
    
    # # Hessian/Jacobian
    # # Initialize matrix
    mat_1 = np.ones((3,leng))

    # First Row
    for i in range(0,leng):
        mat_1[0][i] = X1[1][i] 
    # Second Row
    for i in range(0,leng):
        mat_1[1][i] = X1[2][i]
    # Third Row: all "ones"
    
      #%%
    # Calculations

    F_delta = np.matmul(F.T,mat_1.T)

    # # Gradient
    F_delta_1 = np.matmul(mat_1, mat_1.T)
#     print(F_delta_1)
    
       # Solve coefficients p
    # Gradient inverse
    F_delta_1_inv = np.linalg.pinv(F_delta_1)

    # Solve for p
    p = np.matmul(-F_delta, F_delta_1_inv)
#     print("P:", p)
    # Convert to list
    p_list = p.tolist()
    p_list = p_list[0]
    # coeff.append(p_list[0])

    # Update guesses
    x1 = x1 + p_list[0]
    x2 = x2 + p_list[1]
    x3 = x3 + p_list[2]
    # pp = p.tolist()
    # ppp3 = pp[0]
    # print(type(pp))
    # print(ppp3[0])
    # Calculate new tolerance
    tol = abs(p_list[0])
    tol = abs(np.sum(F_delta))
    coeff.append(tol)
    print(tol)

#     sumofsquares = 0
#     sumofresiduals = 0
    
#     y_mean = np.mean(Y)

#     for i in range(len(Y)):
#         y_pred = x1 * X1[1][i] + x2 * X1[2][i] + x3
#         sumofsquares = (Y[i] - y_mean) ** 2
#         sumofresiduals = (Y[i] - y_pred) ** 2        
#         score = 1 - (sumofresiduals/sumofsquares)
#     print('score: \n', score)
#     if iter > 10000:
#         break


# i = 1e-1
# while i > 1e-5:
#     iter += 1
#     # print(i)
#     i -= 0.0005

# print(iter)
# print("P:", p_list)

Y = []
for i in range(0, leng):
    # y0 = x1 * np.exp(x2 * t[i])
    y0 = (x1 * (X_1[i])) + (x2 * X_2[i]) + x3
    Y.append(y0)

# print(Y)
print('coeff:',coeff)
print('Y:\n',Y)
print('x1:',x1)
print('x2:',x2)
print('x3:',x3)

min_theta = [x3,x2,x1]
min_theta = np.array(min_theta)
min_theta_2 = min_theta.reshape(3,1)
    
    


# In[10]:


# predict the price of a house with 1650 square feet and 3 bedrooms
# add bias unit 1.0
X_predict = np.array([1.0,1650.0,3]) 
#feature scaling the data first
X_predict[1] = (X_predict[1] - mean_size)/ (std_size) 
X_predict[2] = (X_predict[2]- mean_bedroom)/ (std_bedroom)
hypothesis = X_predict.dot(min_theta)
print("Cost of house with 1650 sq ft and 3 bedroom is ",hypothesis)


# In[ ]:





# In[ ]:




