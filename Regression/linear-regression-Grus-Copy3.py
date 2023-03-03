#!/usr/bin/env python
# coding: utf-8

# # Chapter 14 - Linear Regression

# In[1937]:


import os.path
import random
import sys
import numpy as np


# In[1938]:


book_dir = '/Users/CBare/Documents/projects/data-science-from-scratch'
sys.path.extend(os.path.join(book_dir, 'chapter_{:02d}'.format(i)) for i in [3,4,5,6,7,8])


# In[1939]:


# from stats import mean, de_mean, standard_deviation, correlation
# from gradient_descent import minimize_stochastic

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
    data = zip(x, y)
    
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

        if np.all(value) < np.all(min_value):
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


# In[1940]:


import matplotlib.pyplot as plt


# In[1941]:


def predict(alpha, beta, x_i):
    return beta * x_i + alpha


# In[1942]:


# def predict(x_i,theta):
#     return x_i.dot(theta)


# In[1943]:


def error(alpha, beta, x_i, y_i):
    return y_i - predict(alpha, beta, x_i)


# In[1944]:


def sum_of_squared_errors(alpha, beta, x, y):
    return sum(error(alpha, beta, x_i, y_i) ** 2
               for x_i, y_i in zip(x, y))


# In[1945]:


def least_squares_fit(x, y):
    """given training values for x and y,
       find the least-squares values of alpha and beta"""
    beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
    alpha = mean(y) - beta * mean(x)
    return alpha, beta


# In[1946]:


num_friends_good = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


# In[1947]:


daily_minutes_good = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]


# In[1948]:


alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)
print(alpha, beta)


# In[1949]:


y_hat = [predict(alpha, beta, x_i) for x_i in num_friends_good]


# In[1950]:


plt.scatter(num_friends_good, daily_minutes_good, marker='.', color='blue', label='ys1')
plt.plot(num_friends_good, y_hat, '-')
plt.xlabel('friends')
plt.ylabel('minutes')
plt.show()


# In[1951]:


def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))


# In[1952]:


total_sum_of_squares(daily_minutes_good)


# In[1953]:


def r_squared(alpha, beta, x, y):
    """
    the fraction of variation in y captured by the model, which equals
    1 - the fraction of variation in y not captured by the model
    """
    return 1.0 - (sum_of_squared_errors(alpha, beta, x, y) /
                  total_sum_of_squares(y))


# In[1954]:


r_squared(alpha, beta, num_friends_good, daily_minutes_good)


# In[1955]:


# def squared_error(x_i, y_i, theta):
#     alpha, beta = theta
#     return error(alpha, beta, x_i, y_i) ** 2


# In[1956]:


# def squared_error_gradient(x_i, y_i, theta):
#     alpha, beta = theta
#     return [-2 * error(alpha, beta, x_i, y_i),
#             -2 * error(alpha, beta, x_i, y_i) * x_i]


# In[1957]:


# # def sqerror_gradient(x_i, y_i, theta):
# #     alpha,beta = theta
# #     err = error(alpha, beta, x_i, y_i)
# #     return [2 * err * x_i]

# # def error(alpha, beta, x_i, y_i):
# #     return y_i - predict(alpha, beta, x_i)

# def error(x, y, beta):
#     return predict(x, beta) - y

# def sqerror_gradient(x_i, y_i,theta):
#     alpha,beta = theta
#     err = error(alpha, beta, x_i, y_i)
#     return [2 * err * x for x in x_i]


# In[1958]:


def predict2(x, beta):
    """assumes that the first element of x is 1"""
    return np.dot(x, beta)

def error(x, y, beta):
    return predict2(x, beta) - y

def squared_error(x, y, beta):
    return error(x, y, beta) ** 2

def sqerror_gradient(x, y, beta):
    err = error(x, y, beta)
    return [2 * err * x_i for x_i in x]


# In[1959]:


X = np.array(num_friends_good)
Y = np.array(daily_minutes_good)
# print(X)
X_ones = np.ones(len(X))
# X_ones.reshape(len(X),1)

# Add ones to X array
X1 = np.vstack((X_ones,X))


# In[1960]:


random.seed(0)
# theta = np.array([random.random(), random.random()])
Theta = np.linalg.pinv(X1.dot(X1.T)).dot(X1).dot(Y)
# Theta = [random.random(),random.random()]
min_theta = minimize_stochastic(squared_error,
                                  sqerror_gradient,
                                  X,
                                  Y,
                                  Theta,
                                  0.0001)
print(min_theta)


# In[1961]:


y_hat = [predict(x_i,min_theta[0],min_theta[1]) for x_i in np.array(num_friends_good)]
y_hat

# alpha = min_theta[0]
# beta = min_theta[1]
# y_hat = [predict(alpha, beta, x_i) for x_i in num_friends_good]


# In[1962]:


plt.figure(figsize=(10,6))
plt.scatter(num_friends_good, daily_minutes_good, marker='.', color='blue', label='ys1')
plt.plot(num_friends_good,y_hat, '-')
plt.xlabel('friends')
plt.ylabel('minutes')
plt.show()


# In[ ]:


# xx0 = np.linspace(0, max(case), len(case))
# xx1 = np.linspace(0, max(dist), len(dist))
# # yy0 = np.array(b_hat[0] + b_hat[1] * xx0 + b_hat[2]*xx0)
# # yy1 = np.array(b_hat[0] + b_hat[1] * xx1 + b_hat[2]*xx1)
# zz = np.array(b_hat[0] + b_hat[1] * xx1)


# In[ ]:




