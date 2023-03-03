#!/usr/bin/env python
# coding: utf-8

# # Linear Regression using Least Squares
#  
# ![animation](animation-gif2.gif)  
# ## Linear Regression  
# In statistics, linear regression is a linear approach to modelling the relationship between a dependent variable and one or more independent variables. In the case of one independent variable it is called simple linear regression. For more than one independent variable, the process is called mulitple linear regression. We will be dealing with simple linear regression in this tutorial.  
# Let **X** be the independent variable and **Y** be the dependent variable. We will define a linear relationship between these two variables as follows:  
# 
# \\[ Y = mX + c \\]  
# ![mxplusc](http://www.nabla.hr/SlopeInterceptLineEqu.gif)
# 
# This is the equation for a line that you studied in high school. **m** is the slope of the line and **c** is the y intercept. Today we will use this equation to train our model with a given dataset and predict the value of **Y** for any given value of **X**.  
#   
# Our challenege today is to determine the value of **m** and **c**, that gives the minimum error for the given dataset. We will be doing this by using the **Least Squares** method.  
# 
# ## Finding the Error  
# So to minimize the error we need a way to calculate the error in the first place. A **loss function** in machine learning is simply a measure of how different the predicted value is from the actual value.  
# Today we will be using the **Quadratic Loss Function** to calculate the loss or error in our model. It can be defined as: 
#   
# \\[ L(x) = \sum_{i=1}^n (y_i - p_i)^2\\]  
# ![error](error.jpg)   
#   
# We are squaring it because, for the points below the regression line **y - p** will be negative and we don't want negative values in our total error.  
# 
# ## Least Squares method  
# Now that we have determined the loss function, the only thing left to do is minimize it. This is done by finding the partial derivative of **L**, equating it to 0 and then finding an expression for **m** and **c**. After we do the math, we are left with these equations:    
#   
# \\[m = \frac{\sum_{i=1}^n (x_i - \bar x)(y_i - \bar y)}{\sum_{i=1}^n (x_i - \bar x)^2}\\]  
#   
# \\[c = \bar y - m\bar x\\]  
#   
# Here $\bar x$ is the mean of all the values in the input **X** and $\bar y$ is the mean of all the values in the desired output **Y**. This is the Least Squares method. 
# Now we will implement this in python and make predictions.  
# 

# ## Implementing the Model

# In[1]:


# Making imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)


# In[2]:


# Preprocessing Input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()


# In[3]:


data


# In[4]:


# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

num = 0
den = 0
for i in range(len(X)):
    num += (X[i] - X_mean)*(Y[i] - Y_mean)
    den += (X[i] - X_mean)**2
m = num / den
c = Y_mean - m*X_mean

print (m, c)
# X[0] - X_mean


# In[5]:


# Making predictions
Y_pred = m*X + c

plt.scatter(X, Y) # actual
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()

