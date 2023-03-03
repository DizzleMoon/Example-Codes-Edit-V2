#!/usr/bin/env python
# coding: utf-8

# In[80]:


import numpy as np


# In[81]:


def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # y_true and y_pred are numpy arrays of the same length.
    print(y_pred)
    print(y_true)
    return ((y_true - y_pred) ** 2).mean()

def cost(actual,predict):
  m = actual.shape[0]
#   m = actual.shape[1]
  cost__ = -np.sum(np.multiply(np.log(predict), actual) + np.multiply((1 - actual), np.log(1 - predict)))/m
  return np.squeeze(cost__)


# In[82]:


class OurNeuralNetwork:
  '''
  A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
  *** DISCLAIMER ***:
  The code below is intended to be simple and educational, NOT optimal.
  Real neural net code looks nothing like this. DO NOT use this code.
  Instead, read/run it to understand how this specific network works.
  '''
  def __init__(self):
    # Weights
    self.w1 = np.random.normal()
    self.w11 = np.random.normal(2,1)
    self.w2 = np.random.normal()
    self.w22 = np.random.normal(2,1)
    self.w3 = np.random.normal()
    self.w4 = np.random.normal()
    self.w5 = np.random.normal()
    self.w6 = np.random.normal()
    self.w7 = np.random.normal()
    self.w8 = np.random.normal()
    self.w9 = np.random.normal()
    self.w10 = np.random.normal()
    self.w11 = np.random.normal()
    self.w12 = np.random.normal()
    

    # Biases
    self.b1 = np.random.normal()
    self.b2 = np.random.normal()
    self.b3 = np.random.normal()
    self.b4 = np.random.normal()

  def feedforward(self, x):
    # x is a numpy array with 2 elements.
#     h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    w1 = [self.w1,self.w2, self.w7]
    h1 = sigmoid(np.dot(w1,x) + self.b1)
#     print('W1:', self.w1)
#     print('h11:', h1)
#     h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    w2 = [self.w3,self.w4, self.w8]
    h2 = sigmoid(np.dot(w2,x) + self.b2)
    
    w3 = [self.w9,self.w10,self.w11]
    h3 = sigmoid(np.dot(w3,x) + self.b3)
    
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.w12 * h3 + self.b4)
#     print('o1:',o1)
    return np.max(o1)

  def feedforward2(self, x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return np.max(o1)

  def train(self, data, all_y_trues):
    '''
    - data is a (n x 2) numpy array, n = # of samples in the dataset.
    - all_y_trues is a numpy array with n elements.
      Elements in all_y_trues correspond to those in data.
    '''
    learn_rate = 0.1
    epochs = 1000 # number of times to loop through the entire dataset

    for epoch in range(epochs):
      for x, y_true in zip(data, all_y_trues):
        # --- Do a feedforward (we'll need these values later)
        W1 = [self.w1,self.w2,self.w7]
#         W1 = np.array(W1).reshape(1,3)
#         x =  np.array(x).reshape(1,3)
        print('W1:', W1)
        print('x:', x)
        sum_h1 = np.dot(W1,x) + self.b1
#         sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
        h1 = sigmoid(sum_h1)
    
        W2 = [self.w3,self.w4, self.w8]
        sum_h2 = np.dot(W2,x) + self.b2

#         sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h2 = sigmoid(sum_h2)
    
        W3 = [self.w9,self.w10,self.w11]
        sum_h3 = np.dot(W3,x) + self.b3

#         sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
        h3 = sigmoid(sum_h3)

        # 
        sum_o1 = self.w5 * h1 + self.w6 * h2 + self.w12 * h3 + self.b4
        
#         o1 = self.feedforward(x)
        
        o1 = sigmoid(sum_o1)
        y_pred = o1

        # --- Calculate partial derivatives.
        # --- Naming: d_L_d_w1 represents "partial L / partial w1"
#         d_L_d_ypred = -2 * (y_true - y_pred)
#         print(y_true)
        m = y_true + 1
#         print('m_true:', y_true)
#         print('m_pred:', y_pred)
#         d_L_d_ypred = -np.sum(np.multiply(np.log(y_pred), y_true) + np.multiply((1 - y_true), np.log(1 - y_pred)))/m
        d_L_d_ypred = -((y_true - y_pred)/(y_pred*(1 - y_pred)))/m

        # Neuron o1
        d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
        d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
        d_ypred_d_b3 = deriv_sigmoid(sum_o1)

        d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
        d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

        # Neuron h1
        d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
#         print("d_h1_d_w1:", d_h1_d_w1)
#         print('x')
#         d_h1_d_w10 = x * deriv_sigmoid(sum_h1)
#         print("d_h1_d_w10:", d_h1_d_w10)
        d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
        d_h1_d_b1 = deriv_sigmoid(sum_h1)
        
#         d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
#         print("d_h1_d_w1:", d_h1_d_w1)
# #         print('x0:', x)
# #         d_h1_d_w10 = x * deriv_sigmoid(sum_h1)
# #         print("d_h1_d_w10:", d_h1_d_w10)
#         d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
#         print("d_h1_d_w1:", d_h1_d_w2)
#         d_h1_d_w12 = np.dot(x,deriv_sigmoid(sum_h1))
#         print("d_h1_d_w12:", d_h1_d_w12)
#         d_h1_d_b1 = deriv_sigmoid(sum_h1)

        # Neuron h2
        d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
        d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
        d_h2_d_b2 = deriv_sigmoid(sum_h2)

        # --- Update weights and biases
        # Neuron h1
        self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
#         print('W1:', self.w1)
        self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
        self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

        # Neuron h2
        self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
        self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
        self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

        # Neuron o1
        self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
        self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
        self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

      # --- Calculate total loss at the end of each epoch
      if epoch % 10 == 0:
        y_preds = np.apply_along_axis(self.feedforward, 1, data)
#         loss = mse_loss(all_y_trues, y_preds)
        loss = cost(all_y_trues, y_preds)
#         print("Epoch %d loss: %.3f" % (epoch, loss))


# In[83]:


# Define dataset
data = np.array([
  [-2, -1, -4],  # Alice
  [25, 6, 32],   # Bob
  [17, 4, -13],   # Charlie
  [-15, -6, 9], # Diana
])
all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# X=np.array([np.transpose([0,0,1,1]),[0,1,0,1].T,[1,0,0,1].T])
X=np.array([[0,0,1,1],[0,1,1,0],[1,0,0,1]])
# X_xor=np.array([[0,0,1,1],[0,1,1,0]])
X_xor=np.array([[0,0,1,1],[0,1,0,1]])
# x1 = x.copy()
# These are XOR outputs
Y=np.array([0,1,0,1])
# X.reshape(4,3)
# np.transpose([0,0,1,1]).T
X
data_xor = X.T
# print(x1)
# print(data)


# In[84]:


# Train our neural network!
network = OurNeuralNetwork()
# network.train(data_xor, Y)
network.train(data, all_y_trues)


# In[85]:


# Make some predictions
# emily = np.array([-7, -3]) # 128 pounds, 63 inches
# frank = np.array([20, 2])  # 155 pounds, 68 inches
# network.feedforward(emily)
test = np.array([[20],[2], [-6]])
network.feedforward(test)
# print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
# print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M


# In[ ]:




