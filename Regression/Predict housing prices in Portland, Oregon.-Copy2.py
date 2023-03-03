#!/usr/bin/env python
# coding: utf-8

# ## Predict housing prices in Portland, Oregon.
# #### Problem statement :
# Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.
# The file ex1data2.txt contains a training set of housing prices in Portland,Oregon. The first column is the size of the house (in square feet), the second column is the number of bedrooms and the third column is the price of the house.
# Dataset is like below :
# 
# | Size of the house (in square feet) | Number of bedrooms | Price of the house |
# |------------------------------------|--------------------|--------------------|
# | 2104                               | 3                  | 399900             |
# | 1600                               | 3                  | 329900             |
# | 2400                               | 3                  | 369000             |
# 
# Now we have to predict housing prices in Portland, Oregon.(including which is not mention in our example dataset).
# 
# **Note:This problem statement and dataset is from coursera Andrew ng machine learning [Coursework](https://www.coursera.org/learn/machine-learning)**

# In[25]:


import numpy as np
import matplotlib.pyplot as plt
#supressing the scientific output
np.set_printoptions(suppress=True) 


# In[26]:


data = np.loadtxt("ex1data2.txt",dtype=np.float64,delimiter=",")
data[:5,::] #dataset loaded demonstration


# In[27]:


# Break datasets into X and Y.
X = data[::,0:2]
Y = data[::,-1:]


# In[28]:


# Plotting example dataset
plt.figure(figsize = (15,4),dpi=100)
plt.subplot(121)
plt.scatter(X[::,0:1],Y)
plt.xlabel("Size of house (X1)")
plt.ylabel("Price (Y)")
plt.subplot(122)
plt.scatter(X[::,-1:],Y)
plt.xlabel("Number of Bedrooms (X2)")
plt.ylabel("Price (Y)")
plt.show()


# In[29]:


# introduce weights of hypothesis (randomly initialize)
Theta = np.random.rand(1,3)
Theta_0 = np.array([np.random.random(),np.random.random(),np.random.random()])
Theta_1 = np.matrix(Theta_0)
print("Theta:", Theta)
print("Theta_0:", Theta_1)
# m is total example set , n is number of features
m,n = X.shape
# add bias to input matrix by simple make X0 = 1 for all
X_bias = np.ones((m,n+1))
X_bias[::,1:] = X
# output first 5 X_bias examples
print("X_bias = \n",X_bias[0:5,:])
print("Y = \n",Y[0:5,::])


# In[30]:


theta_0 = np.linalg.pinv(X_bias.dot(X_bias.T))
theta_1 = theta_0.dot(X_bias)
theta_2 = theta_1.T.dot(Y)
theta_2

# Theta = np.linalg.pinv(X_bias.dot(X_bias.T)).dot(X_bias.T).dot(Y.T)
# Theta


# In[31]:


#feature scaling
# it also protect program from overflow error
mean_size = np.mean(X_bias[::,1:2])
mean_bedroom = np.mean(X_bias[::,2:])
size_std = np.std(X_bias[::,1:2])
bedroom_std = np.std(X_bias[::,2:])
X_bias[::,1:2] = (X_bias[::,1:2] - mean_size)/ (size_std) 
X_bias[::,2:] = (X_bias[::,2:] - mean_bedroom)/ (bedroom_std)
X_bias[0:5,::]


# In[32]:


#define function to find cost
def cost(X_bias,Y,Theta):
#     np.seterr(over='raise')
    m,n = X_bias.shape
    hypothesis = X_bias.dot(Theta.transpose())
    return (1/(2.0*m))*((np.square(hypothesis-Y)).sum(axis=0))


# In[33]:


#function gradient descent algorithm from minimizing theta
def gradientDescent(X_bias,Y,Theta,iterations,alpha):
    count = 1
    cost_log = np.array([])
    while(count <= iterations):
        hypothesis = X_bias.dot(Theta.transpose())
        temp0 = Theta[0,0] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,0:1])).sum(axis=0)
        temp1 = Theta[0,1] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,1:2])).sum(axis=0)
        temp2 = Theta[0,2] - alpha*(1.0/m)*((hypothesis-Y)*(X_bias[::,-1:])).sum(axis=0)
        Theta[0,0] = temp0
        Theta[0,1] = temp1
        Theta[0,2] = temp2
        cost_log = np.append(cost_log,cost(X_bias,Y,Theta))
        count = count + 1
    plt.plot(np.linspace(1,iterations,iterations,endpoint=True),cost_log)
    plt.title("Iteration vs Cost graph ")
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost of Theta")
    plt.show()
    return Theta


# In[34]:


alpha = 0.0003
iterations = 100
Theta = gradientDescent(X_bias,Y,theta_2.T,iterations,alpha)
print(Theta)


# In[35]:


# predict the price of a house with 1650 square feet and 3 bedrooms
# add bias unit 1.0
X_predict = np.array([1.0,1650.0,3]) 
#feature scaling the data first
X_predict[1] = (X_predict[1] - mean_size)/ (size_std) 
X_predict[2] = (X_predict[2]- mean_bedroom)/ (bedroom_std)
hypothesis = X_predict.dot(Theta.transpose())
print("Cost of house with 1650 sq ft and 3 bedroom is ",hypothesis)


# In[36]:


sumofsquares = 0
sumofresiduals = 0

# y_hat = [predict(x_i,min_theta_2) for x_i in X1.T]
Y1 = (Y - np.mean(Y,axis=0))/np.std(Y,axis=0)
y_mean = np.mean(Y1)

# for i in X1.T:
#     y_pred = predict(x_i,min_theta_2)
#     sumofsquares += (Y[])
# for x_i in X1.T:
#     print(x_i)
# X12 = X1.T
y_pred = X_bias.dot(Theta.T)
for i in range(len(Y)):
#     X_pred = np.array(X_predict[i])
#     y_pred = X_bias.dot(Theta.T)
#     print(y_pred)
#     y_pred = Theta[0] + Theta[1]*X_bias
    y_pred = X_predict.dot(Theta.T)
    sumofsquares += (Y[i] - y_mean) ** 2
    sumofresiduals += (Y[i] - y_pred) ** 2
#     score = 1 - (sumofresiduals/sumofsquares)
#     print(score[0])
    
score = 1 - (sumofresiduals/sumofsquares)
# print(np.abs(np.mean(score)))
print(score)


# In[ ]:




