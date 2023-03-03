#!/usr/bin/env python
# coding: utf-8

# In[151]:


import os.path
import random
import sys
import numpy as np
from stats import mean, de_mean, standard_deviation, correlation
from gradient_descent import minimize_stochastic
import matplotlib.pyplot as plt


# In[152]:


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
    while iterations_with_no_improvement < 1500:
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
        for x_i, y_i in data:
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))

    return min_theta


# In[153]:


data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")
data[:5,::] #dataset loaded demonstration


# In[154]:


# Break datasets into X and Y.
X_0 = np.array(data[::,0:2])
X_1 = np.array(data[::,0])
X_2 = np.array(data[::,1])
Y = data[::,-1:]
X_10 = np.sort(X_1)
X_10 = X_10[::-1]
X_10 = X_10.tolist()
# X_10
Y


# In[155]:


# Setup bias array
X_ones = np.ones(len(X_0))
# Concentate arrays
# X = [X_ones,X_0]
X1 = np.vstack((X_ones.T,X_0.T))
# X1 = [[X_ones],[X_0]]

# X1.T


# In[156]:


Theta = np.linalg.pinv(X1.dot(X1.T)).dot(X1).dot(Y)
Theta


# In[157]:


theta_0 = np.linalg.pinv(X1.dot(X1.T))
theta_1 = theta_0.dot(X1)
theta_2 = theta_1.dot(Y)
theta_2


# In[158]:


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
print(mean_size)


# In[159]:


def predict(x, beta):
    """assumes that the first element of x is 1"""
#     print("X: \n", x)
    return np.dot(x, beta)


# In[160]:


def error(x, y, beta):
    return predict(x, beta) - y

def squared_error(x, y, beta):
    return error(x, y, beta) ** 2

def sqerror_gradient(x, y, beta):
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]

def cost(X_bias,Y,Theta):
#     np.seterr(over='raise')
    m = X_bias.shape
    Theta = np.array(Theta)
    hypothesis = X_bias.dot(Theta)
    cost_val = (1/(2*m[0]))*((np.square(hypothesis-Y)).sum(axis=0))
    return cost_val


# In[161]:


def minimize_stochastic_v5(target_fn, gradient_fn, x, y, theta_0, alpha_0):
    data = list(zip(x, y))
    
    # initial guess
    theta = theta_0
    
    # initial step size
    alpha = alpha_0 
    
    # the minimum so far
    min_theta, min_value = None, float("inf")
    iterations_with_no_improvement = 0
    
    tol = 1e-2
    min_theta_1 = 0
    cnt = 0
    
    tol2 = []

    # if we ever go 100 iterations with no improvement, stop
    while iterations_with_no_improvement < 1500:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # if we've found a new minimum, remember it
            # and go back to the original step size
            min_theta, min_value = theta, value
#             tol1 = min_theta_1 - np.sum(min_theta)
#             tol = abs(tol1)
#             print('Tol: \n',tol)
    #         print(tol)
    #         min_theta_1 = min_theta[0][0]
#             min_theta_1 = np.sum(min_theta)
            iterations_with_no_improvement -= 1
            alpha = alpha_0
        else:
            # otherwise we're not improving, so try shrinking the step size
            iterations_with_no_improvement += 1
            alpha *= 0.9
            
#         print('alpha:\n',alpha)

        # and take a gradient step for each of the data points
        for x_i, y_i in data:
            gradient_i = gradient_fn(x_i, y_i, theta)
            theta = vector_subtract(theta, scalar_multiply(alpha, gradient_i))
#             cost_val = cost(x_i,y,theta)
            
            
#         cost_val = cost(x,y,theta)
#         print('Cost Val', cost_val)
            
#         Y_err_1 = theta_0 - min_theta
#         Y_err_1_sq = np.square(Y_err_1)
#         tol = np.sum(Y_err_1_sq)
#         tol = squared_error(x_i, y_i, min_theta)
#         print('Y_error:', tol)
            
        cnt += 1
            
#         tol1 = (min_theta_1 - np.sum(min_theta))
#         print('tol1: \n',tol)
#         tol2.append(tol1)
        

    return min_theta,cnt


# In[162]:


# Theta_0 = np.array([1,0,0])
Theta_0 = np.random.rand(1,3)
min_theta,cnt = minimize_stochastic_v5(squared_error,
                                  sqerror_gradient,
                                  X1.T,
                                  Y,
                                  Theta,
                                  0.01)
print(min_theta)
print(cnt)


# In[163]:


# min_theta = minimize_stochastic(squared_error,
#                                   sqerror_gradient,
#                                   X1.T,
#                                   Y,
#                                   Theta,
#                                   0.3)
# min_theta


# In[164]:


xx_i = []
for x_i in X1.T:
    xx_i.append(x_i)
    
print(X1.shape)

y_hat = [predict(x_i,min_theta) for x_i in X1.T]
y_hat = np.sort(y_hat)
np.max(y_hat)
# y_hat
# y_hat_alt = []
# for i in range(len(y_hat)):
#     y_hat_alt.append(y_hat[i][0])
# yy2 = np.sort()
# print(yy2)
# y_hat_alt


# In[165]:


# predict the price of a house with 1650 square feet and 3 bedrooms
# add bias unit 1.0
X_predict = np.array([1.0,1650.0,3]) 
#feature scaling the data first
X_predict[1] = (X_predict[1] - mean_size)/ (std_size) 
X_predict[2] = (X_predict[2]- mean_bedroom)/ (std_bedroom)
hypothesis = X_predict.dot(min_theta)
print("Cost of house with 1650 sq ft and 3 bedroom is ",hypothesis)
# print(mean_size)


# In[166]:


# predict the price of a house with 1650 square feet and 3 bedrooms
# add bias unit 1.0
X_predict = np.array([1.0,1650.0,3]) 
#feature scaling the data first
# X_predict[1] = (X_predict[1] - mean_size)/ (std_size) 
# X_predict[2] = (X_predict[2]- mean_bedroom)/ (std_bedroom)
hypothesis = X_predict.dot(theta_2)
print("Cost of house with 1650 sq ft and 3 bedroom is ",hypothesis)
# print(mean_size)


# In[167]:


def predict_line(alpha, beta, x_i):
    return beta * x_i + alpha


# In[168]:


y_hat_line = [predict_line(min_theta[0],min_theta[1],x_i) for x_i in X11[1]]
# y_hat_line = [predict(x_i, min_theta) for x_i in X11[1]]
y_hat_line
plt.scatter(X_1,Y)
plt.plot(X_1,y_hat_line)


# In[169]:


y_hat_line = [predict_line(min_theta[0],min_theta[2],x_i) for x_i in X11[2]]
# y_hat_line = [predict(x_i, min_theta) for x_i in X11[1]]
y_hat_line
plt.scatter(X_2,Y)
plt.plot(X_2,y_hat_line)


# In[170]:


# Building the model
# X1 = X11[1]
# X_mean_1 = np.mean(X1)
# X2 = X11[2]
# X_mean_2 = np.mean(X2)
# # Y1 = (Y - np.mean(Y,axis=0))/np.std(Y,axis=0)
# Y_mean = np.mean(Y)

# num1 = 0
# den1 = 0
# num2 = 0
# den2 = 0
# for i in range(len(X1)):
#     num1 += (X1[i] - X_mean_1)*(Y[i] - Y_mean)
#     den1 += (X1[i] - X_mean_1)**2
#     num2 += (X2[i] - X_mean_2)*(Y[i] - Y_mean)
#     den2 += (X2[i] - X_mean_2)**2
# m1 = num1 / den1
# m2 = num2 / den2
# c1 = Y_mean - m1*X_mean_1 - m2*X_mean_2

# print(m1, c1)


# # Making predictions
# Y_pred_1 = m1*X1 + c1
# Y_pred_2 = m2*X2 + c1

y_hat_line_1 = [predict_line(min_theta[0],min_theta[1],x_i) for x_i in X11[1]]
plt.scatter(X_1, Y) # actual
plt.plot([min(X_1), max(X_1)], [min(y_hat_line_1), max(y_hat_line_1)], color='red') # predicted
# plt.plot(X,Y_pred)
plt.show()


y_hat_line_2 = [predict_line(min_theta[0],min_theta[1],x_i) for x_i in X11[2]]
plt.scatter(X_2, Y) # actual
plt.plot([min(X_2), max(X_2)], [min(y_hat_line_2), max(y_hat_line_2)], color='green') # predicted
# plt.plot(X,Y_pred)
plt.show()


# In[171]:


sumofsquares = 0
sumofresiduals = 0
score = 0

# y_hat = [predict(x_i,min_theta_2) for x_i in X1.T]

Y1 = (Y - np.mean(Y,axis=0))/np.std(Y,axis=0)

y_mean = np.mean(Y1)
min_theta_2 = np.array(min_theta)

# for i in X1.T:
#     y_pred = predict(x_i,min_theta_2)
#     sumofsquares += (Y[])
# for x_i in X1.T:
#     print(x_i)
# X12 = X1.T
# y_pred = X1.T.dot(min_theta_2)
for i in range(len(Y)):
#     X_pred = np.array(X_predict[i])
#     y_pred = X1.T.dot(min_theta_2)
    y_pred = X_predict.dot(min_theta)
#     print(y_pred)
#     y_pred = Theta[0] + Theta[1]*X_bias
    sumofsquares += (Y[i] - y_mean) ** 2
    sumofresiduals += (Y[i] - y_pred) ** 2
#     score += 1 - np.mean(sumofresiduals/sumofsquares)
#     print(score[0])
    
score = 1 - (sumofresiduals/sumofsquares)
# print(np.abs(np.mean(score)))
# print(np.mean(score))
# print(np.mean(score))
print(score)


# In[172]:


# sumofsquares = 0
# sumofresiduals = 0

# # y_hat = [predict(x_i,min_theta_2) for x_i in X1.T]
# y_mean = np.mean(Y)

# # for i in X1.T:
# #     y_pred = predict(x_i,min_theta_2)
# #     sumofsquares += (Y[])
# # for x_i in X1.T:
# #     print(x_i)
# X12 = X1.T
# min_theta = np.array(min_theta)
# for i in range(len(Y)):
#     y_pred = predict(X12[i],min_theta)
# #     y_pred = X_predict.dot(min_theta)
#     sumofsquares = (Y[i] - y_mean) ** 2
#     sumofresiduals = (Y[i] - y_pred) ** 2
#     score = 1 - (sumofresiduals/sumofsquares)
# #     print(score[0])
    
# # score = 1 - (sumofresiduals/sumofsquares)
# print(score[0])


# In[173]:


# sumofsquares = 0
# sumofresiduals = 0

# # y_hat = [predict(x_i,min_theta_2) for x_i in X1.T]
# y_mean = np.mean(Y)

# # for i in X1.T:
# #     y_pred = predict(x_i,min_theta_2)
# #     sumofsquares += (Y[])
# # for x_i in X1.T:
# #     print(x_i)
# X12 = X1.T
# min_theta = np.array(min_theta)
# for i in range(len(Y)):
#     y_pred = predict(X12[i],min_theta)
# #     y_pred = X_predict.dot(min_theta)
#     sumofsquares = (Y[i] - y_mean) ** 2
#     sumofresiduals = (Y[i] - y_pred) ** 2
#     score = 1 - (sumofresiduals/sumofsquares)
# #     print(score[0])
    
# # score = 1 - (sumofresiduals/sumofsquares)
# print(score[0])


# In[174]:


X12 = X1.T
print(X12)
y_pred = predict(X12,min_theta)
y_pred
# X12[0]


# In[175]:


sumofsquares = (Y - y_mean) ** 2
sumofresiduals = (Y - y_pred) ** 2
score = 1 - (sumofresiduals/sumofsquares)
score.mean()


# In[176]:


# sumofresiduals
# sumofsquares 


# In[177]:


yym = y_mean * np.ones((len(Y),1))
sumofsquares = (Y - yym) ** 2
# sumofsquares


# In[178]:


# score = 1 - (sumofresiduals/sumofsquares)
# score.mean()


# In[179]:


ssr = sumofresiduals.T.dot(sumofresiduals)
ssq = sumofsquares.T.dot(sumofsquares)
ssr


# In[180]:


score = 1 - (ssr/ssq)
score[0]


# In[ ]:




