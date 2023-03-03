#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import math
import matplotlib.pyplot as plt
# np.random.seed(0)


# In[ ]:





# In[3]:


def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
  return ((y_true - y_pred) ** 2).mean()

def cost(actual,predict):
#   m = 4
  m = actual.shape[0]
  print('m_cost:', m)
  cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
  return np.squeeze(cost__)

def feedforward0(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(w1 * x[0] + w2 * x[1] + w3 * x[2] + b1)
    h2 = sigmoid(w4 * x[0] + w5 * x[1] + w6 * x[2] + b2)
    o1 = sigmoid(w7 * h1 + w8 * h2 + b3)
    return o1

def feedforward2(W1,W2,w7,w8,b1,b2,b3,x,data):
    # x is a numpy array with 2 elements.
    # x1
#     np.random.seed(data.shape[0]*data.shape[1])
#     np.random.seed(8)
    x = x.reshape(data.shape[1],1)
    h1 = sigmoid(np.dot(W1,x) + b1)
    h2 = sigmoid(np.dot(W2,x) + b2)
#     h3 = sigmoid(np.dot(W3,x) + b3)
    o1 = sigmoid(w7 * h1 + w8 * h2 + b3)
    return o1

def feedforward3(W1,W2,W3,w7,w8,w9,b1,b2,b3,b4,x,data):
    # x is a numpy array with 2 elements.
    # x1
    x = x.reshape(data.shape[0],1)
    h1 = sigmoid(np.dot(W1,x) + b1)
    h2 = sigmoid(np.dot(W2,x) + b2)
    h3 = sigmoid(np.dot(W3,x) + b3)
    o1 = sigmoid(w7 * h1 + w8 * h2 + w9 * h3 + b4)
    return o1

def feedforward(wg,wg_out,bias,data,X_xor,x):
    h_lst = []
    summation = []
    sigmoid_sum = []
    # x1
    x = x.reshape(data.shape[1],1)

    for i in range(0,len(X_xor)):
        sum_h3 = np.dot(wg[i],x) + bias[0][i]
        summation.append(sum_h3)
        h3 = sigmoid(sum_h3)
        sigmoid_sum.append(h3)
        h1_2 = wg_out[i] * h3
        h_lst.append(h1_2)

    sum_o1 = np.sum(h_lst) + bias[0][-1]

#         print('sum_o1:', sum_o1)
    o1 = sigmoid(sum_o1)
    return o1


# In[4]:


X_xor_2 = 0


# In[5]:


# 1
# X_xor=np.array([[0,1,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(0)
# bias_max = data.shape[1] * data.shape[0]
# if len(X_xor) % data.shape[1] == 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 2
# X_xor=np.array([[0,0,1,1],[0,1,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1]
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0]
# # data.shape
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 


# print(len(X_xor) % (data.shape[1]))
# 4 - Version 2
# X_xor=np.array([[0,0,1,1],[0,0,0,0],[1,0,1,0],[1,1,1,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# bias_max = data.shape[1] * data.shape[0]
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 3
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# bias_max = data.shape[1] + data.shape[0]
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 4
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# bias_max = data.shape[1] + data.shape[0]
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 5
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# bias_max = data.shape[1] + data.shape[0]
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 6
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
# print('Coeff:', len(X_xor) % 2)

# 7
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] + 1
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 8 
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 7
# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] + 1
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 


# 8
# X_xor=np.array([[0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# X_xor=np.array([[0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,0,1],[0,1,0,0],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0]  

# 9
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# X_xor=np.array([[0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0]

# X_xor=np.array([[0,0,1,1],[1,1,0,0],[0,1,0,1],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 


# 10
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 11
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 


# 12
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
    
# 13
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
    
# 14    
# X_xor=np.array([[0,0,1,1],[0,1,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# 15
# X_xor=np.array([[0,0,1,1],[1,1,1,1],[0,1,1,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[0,1,0,0],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 


# 16
# X_xor=np.array([[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
    
# X_xor=np.array([[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 

# X_xor=np.array([[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# # X_xor = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[0,0,1,0],[0,1,1,0],[1,1,1,0],[1,1,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,0,1,1],[1,1,1,0],[1,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0]

# X_xor=np.array([[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
    
    
# 17
# X_xor=np.array([[0,0,0,0],[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# # X_xor = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[0,0,1,0],[0,1,1,0],[1,1,1,0],[1,1,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,0,1,1],[1,1,1,0],[1,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0]

# 18
X_xor=np.array([[1,0,1,1],[0,0,0,0],[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# X_xor = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[0,0,1,0],[0,1,1,0],[1,1,1,0],[1,1,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,0,1,1],[1,1,1,0],[1,1,0,0]])
all_y_trues=np.array([0,1,1,0])
data = X_xor.T
data.shape[0]
seed_num = data.shape[0] * data.shape[1] 
np.random.seed(seed_num)
# bias_max = data.shape[1] + data.shape[0] 
if len(X_xor) % data.shape[1]== 0:
    bias_max = data.shape[1] + data.shape[0] + 1
else:
    bias_max = data.shape[1] + data.shape[0]

# 19
# X_xor=np.array([[1,0,1,1],[0,0,0,1],[0,0,0,0],[0,0,1,1],[0,0,0,1],[1,1,1,1],[0,1,0,0],[0,1,1,0],[1,0,1,0],[1,1,1,0],[1,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,1],[1,1,0,1],[0,1,1,0],[1,0,0,0],[0,0,0,1]])
# # X_xor = np.array([[0,0,0,0],[0,0,0,1],[0,0,1,1],[0,1,1,1],[1,1,1,1],[0,0,1,0],[0,1,1,0],[1,1,1,0],[1,1,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,0,1,1],[1,1,1,0],[1,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 
# np.random.seed(seed_num)
# # bias_max = data.shape[1] + data.shape[0] 
# if len(X_xor) % data.shape[1]== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0]



# print('Coeff:', len(X_xor) % 3)

# data = np.array([
#   [-2, -1],  # Alice
#   [25, 6],   # Bob
#   [17, 4],   # Charlie
#   [-15, -6], # Diana
# ])
# all_y_trues = np.array([
#   1, # Alice
#   0, # Bob
#   0, # Charlie
#   1, # Diana
# ])
# seed_num = data.shape[0] * data.shape[1]
# np.random.seed(seed_num)
# X_xor = data.T
# X_xor_2 = 'ABC'
# if len(X_xor) % (data.shape[1])== 0:
#     bias_max = data.shape[1] + data.shape[0] + 1
# else:
#     bias_max = data.shape[1] + data.shape[0] 
# bias_max = data.shape[0] + data.shape[1]

# # data.shape[1]
# print(data.shape[0])
# print(data.shape[1])

# print(len(X_xor) % (data.shape[0]))
# print(len(X_xor) % (data.shape[1]))

len(X_xor) % data.shape[1]
# data.shape[1]
# (data.shape[1]+1) % 10


# In[ ]:





# In[6]:


# np.random.seed(12)

# Weights
# w7 = np.random.normal()
# w8 = np.random.normal()
# w9 = np.random.normal()
# wg_out = [w7,w8,w9]

# Biases
# b1 = np.random.normal()
# b2 = np.random.normal()
# b3 = np.random.normal()
# b4 = np.random.normal()


# In[7]:


# learn_rate_wg = 2.2
# learn_rate_bias = 2.8
# epochs = 1000 # number of times to loop through the entire dataset


# In[8]:


# Weights
# hidden_layer_size = (len(X_xor) % data.shape[0])
# print('HL_1:',hidden_layer_size )

# hidden_layer_size = math.floor(len(X_xor) * 0.5)
# print('HL_2:',hidden_layer_size)
# # print(len(X_xor) % data.shape[0])

# if hidden_layer_size > len(X_xor):
hidden_layer_size = math.floor(len(X_xor) * 0.5)
print(hidden_layer_size)
print(len(X_xor) % data.shape[0])

if hidden_layer_size <= 1:
    hidden_layer_size += len(X_xor) % data.shape[0] 
    
# wg = np.random.normal(size=(len(X_xor) +(data.shape[1] + data.shape[0]) - hidden_layer_size,data.shape[1]))
wg = np.random.normal(size=(len(X_xor) + (data.shape[1] + data.shape[0]) - hidden_layer_size,data.shape[1]))
# wg = np.squeeze(wg)
# print('wg:', wg)
if len(X_xor) == 1:
    wg_out = np.random.normal(size=(2,data.shape[1]))
else:    
    wg_out = np.random.normal(size=(1,data.shape[1]))
    wg_out = np.squeeze(wg_out)

# w7 = wg_out[0]
# w8 = wg_out[1]
# w9 = wg_out[2]

# Weights
bias = np.random.normal(size=(1,bias_max))
# bias
# print('wg:',wg)
# print(wg[0][0])
# print(bias)
# print(bias[0][0])
# wg_out[0][1]
# wg_out

# data.shape[0]

# wg_out = wg_out.tolist()
wg_out


# In[9]:


def train_fit(wg,wg_out,bias,data,all_y_trues):

    learn_rate_wg = 1.9
    learn_rate_bias = 2.8
    
#     if (data.shape[1]) % 9 == 1 or (data.shape[1]+1) % 10 == 0:
#         learn_rate_wg = learn_rate_wg + data.shape[1]
    if (data.shape[1]) % (data.shape[0] + data.shape[1]) == 9 and (data.shape[1]+1) % (data.shape[0] + data.shape[1]) == 10:
        learn_rate_wg = learn_rate_wg + data.shape[1]
        learn_rate_bias = learn_rate_bias + bias.shape[1]
    if (data.shape[1]) % (data.shape[0] + data.shape[1]) == 2 and (data.shape[1]+1) % (data.shape[0] + data.shape[1]) == 3:
        learn_rate_wg = learn_rate_wg + data.shape[1]
        learn_rate_bias = learn_rate_bias + bias.shape[1]
#         learn_rate_bias = learn_rate_bias + bias.shape[1]
#     if data.shape[0] % 10 == 4:
#         learn_rate_wg = learn_rate_wg + data.shape[0]
        
#     else:
#         learn_rate_wg = learn_rate_wg + data.shape[0]
    
#     learn_rate_wg = 1.9
#     learn_rate_bias = 2.8
    epochs = 1000 # number of times to loop through the entire dataset  
    mse_0 = 0.1
    mse = 0.2
    losses = []
    
    for epoch in range(epochs):
                    
#         if mse < mse_0:
#             learn_rate_wg *= 0.99
#             learn_rate_bias *= 0.99
#         else:
#             learn_rate_wg *= 0.001
#             learn_rate_bias *= 0.001

#         mse_0 = mse            
                
        
        for x, y_true in zip(data, all_y_trues):
            # --- Do a feedforward (we'll need these values later)
            h_lst = []
            summation = []
            sigmoid_sum = []
            # x1
            x = x.reshape(data.shape[1],1)


            for i in range(0,len(X_xor)):
                sum_h3 = np.dot(wg[i],x) + bias[0][i]
                summation.append(sum_h3)
                h3 = sigmoid(sum_h3)
                sigmoid_sum.append(h3)
                h1_2 = wg_out[i] * h3
                h_lst.append(h1_2)

            sum_o1 = np.sum(h_lst) + bias[0][-1]

    #         print('sum_o1:', sum_o1)
            o1 = sigmoid(sum_o1)
            y_pred = o1
#             print(y_pred)

    #         m = y_true + data.shape[0]
            m = 4
    #             print('m:', m)
            d_L_d_ypred = ((y_pred - y_true)/(y_pred*(1 - y_pred))).mean()
#             print(d_L_d_ypred)
#             if abs(d_L_d_ypred) >= 1.05:
#                 learn_rate_wg *= 1.05
#                 learn_rate_bias *= 0.01
#             elif abs(d_L_d_ypred) <= 0.5:
#                 learn_rate_wg *= 0.05
#                 learn_rate_bias *= 0.05
                
#             mse = ((y_true - y_pred) ** 2).mean()
#             print('mse:', mse)
#             mse = -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred)).mean()
            mse = -np.sum(np.multiply(np.log(y_pred), y_true) + np.multiply((1 - y_true), np.log(1 - y_pred))).mean()
#             losses.append(mse)
            
#             if mse > 0.0001:
#                 learn_rate_wg *= 1.0085
#                 learn_rate_bias *= 1.005
#             else:
#                 learn_rate_wg *= 0.001
#                 learn_rate_bias *= 0.01
            mse_power = data.shape[0] + data.shape[1] + 1
            if mse > 1/(10**(mse_power)):
                learn_rate_wg *= 1 + (0.5 * (10 ** -(data.shape[0])))
                learn_rate_bias *= 1 + (0.5 * (10 ** -(bias.shape[1])))
            else:
#                 learn_rate_wg *= 0.01
#                 learn_rate_bias *= 0.01
                learn_rate_wg *= 10 ** -data.shape[0]
                learn_rate_bias *= 10 ** -bias.shape[1]

#             if mse > 0.0000001:
#                 learn_rate_wg *= 1.05
#                 learn_rate_bias *= 1.05
#             else:
#                 learn_rate_wg *= 0.001
#                 learn_rate_bias *= 0.001
                
#             mse_0 = mse     

#             if mse < mse_0:
#                 learn_rate_wg *= 1.99
#                 learn_rate_bias *= 0.99
#             else:
#                 learn_rate_wg *= 0.009
#                 learn_rate_bias *= 0.001

#             mse_0 = mse   

                
                
            d_ypred_d_b4 = deriv_sigmoid(sum_o1).mean()

            d_ypred_d_w_out = []
            for i in range(len(X_xor)):
                d_ypred_d_w_out.append(sigmoid_sum[i] * deriv_sigmoid(sum_o1))

            d_ypred_d_h = []
            for i in range(len(summation)):
                d_ypred_d_h.append(np.dot(wg_out[i],deriv_sigmoid(sum_o1)))

            d_h1_d_b = []
            for i in range(len(summation)):
                d_h1_d_b.append(deriv_sigmoid(summation[i]).mean())

            neuron = []
            for i in range(len(summation)):
                neuron.append(np.dot(x,deriv_sigmoid(summation[i]).mean()))

    #         print(neuron)

            for i in range(len(summation)):
    #             bias[0][i] -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h[i] * d_h1_d_b[i]
                bias[0][i]  -= learn_rate_bias * d_L_d_ypred * np.dot(d_ypred_d_h[i],d_h1_d_b[i])

            for n in range(len(neuron)):
                for i in range(len(summation)):
                    wg[n][i] -= learn_rate_wg * d_L_d_ypred * np.dot(d_ypred_d_h[i],neuron[n][i])

            for i in range(len(summation)):
                wg_out[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w_out[i]


            bias[0][-1] -= learn_rate_bias * d_L_d_ypred * d_ypred_d_b4
            
        losses.append(mse)

        
            
    return wg,wg_out,bias,losses
            
        
print(data.shape[1] % 9)
print((data.shape[1]+1) % 9)
(data.shape[1]+1) % 10
print((data.shape[1]) % (data.shape[0] + data.shape[1]))
print((data.shape[1]+1) % (data.shape[0] + data.shape[1]))


# In[10]:


wg,wg_out,bias,losses = train_fit(wg,wg_out,bias,data,all_y_trues)
# plt.plot(losses)


# In[11]:


# W1 = wg[0]
# W2 = wg[1]
# W3 = wg[2]

# # w7 = wg_out[0]
# # w8 = wg_out[1]
# # w9 = wg_out[2]

# w7 = wg_out[0]
# w8 = wg_out[1]
# # w9 = wg_out[2]



# b1 = bias[0][0]
# b2 = bias[0][1]
# b3 = bias[0][-1]
# # b4 = bias[0][-1]


# In[12]:


# test = np.array([1])
# pred = feedforward(wg,wg_out,bias,data,X_xor,test)
# pred


# In[13]:


# # Make some predictions
if len(X_xor) == 3:
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward3(W1,W2,W3,w7,w8,w9,b1,b2,b3,b4,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0])  # 155 pAounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward3(W1,W2,W3,w7,w8,w9,b1,b2,b3,b4,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward3(W1,W2,W3,w7,w8,w9,b1,b2,b3,b4,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward3(W1,W2,W3,w7,w8,w9,b1,b2,b3,b4,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[14]:


# # Make some predictions
if len(X_xor) == 2 and X_xor_2 != 'ABC':
    np.random.seed(seed_num)
    test = np.array([1, 0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0])  # 155 pAounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
# elif X_xor_2 == 'ABC' and len(X_xor) == 2:
#     #     # Make some predictions
#     emily = np.array([-7, -3]) # 128 pounds, 63 inches
#     frank = np.array([20, 2])  # 155 pounds, 68 inches
#     print("Emily: %.7f" % feedforward(wg,wg_out,bias,data,X_xor,emily)) # 0.951 - F
#     print("Frank: %.7f" % feedforward(wg,wg_out,bias,data,X_xor,frank)) # 0.039 - M
#     # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
#     # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[15]:


# # # Make some predictions
# if len(X_xor) == 1:
#     test = np.array([0]) # 128 pounds, 63 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     test = np.array([1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     test = np.array([1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     test = np.array([0])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
#     # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
#     # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[16]:


# # Make some predictions
if len(X_xor) == 1:
    test = np.array([0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[17]:


if X_xor_2 == 'ABC':
    #     # Make some predictions
    emily = np.array([-7, -3]) # 128 pounds, 63 inches
    frank = np.array([20, 2])  # 155 pounds, 68 inches
    print("Emily: %.7f" % feedforward(wg,wg_out,bias,data,X_xor,emily)) # 0.951 - F
    print("Frank: %.7f" % feedforward(wg,wg_out,bias,data,X_xor,frank)) # 0.039 - M
    # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[18]:


if len(X_xor) == 4:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[19]:


if len(X_xor) == 5:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
    np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[20]:


if len(X_xor) == 6:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[21]:


if len(X_xor) == 7:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[22]:


if len(X_xor) == 8:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[23]:


if len(X_xor) == 9:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[24]:


if len(X_xor) == 10:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[25]:


if len(X_xor) == 11:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[26]:


if len(X_xor) == 12:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))


# In[27]:


if len(X_xor) == 13:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[28]:


if len(X_xor) == 14:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[29]:


if len(X_xor) == 15:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[30]:


if len(X_xor) == 16:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1,1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0,0,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[31]:


if len(X_xor) == 17:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1,1,0,1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0,0,1,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[32]:


if len(X_xor) == 18:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1,1,0,1,0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0,0,1,0,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])


# In[33]:


if len(X_xor) == 19:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
#     np.random.seed(seed_num)
    test = np.array([1, 0, 1, 1,0, 1,0,0,1,0,1,0,0,1,1,0,1,0,1]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([0, 0, 0, 0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 0, 1, 0, 1,0,1,1,1,0,0,1,0,0,0,1,0,1,0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])
    test = np.array([1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    print(feedforward(wg,wg_out,bias,data,X_xor,test))
#     print([feedforward(wg,wg_out,bias,data,X_xor,test) for _ in range(2)])

