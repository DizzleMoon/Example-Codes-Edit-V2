#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
np.random.seed(12)


# In[2]:


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

def feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,x):
    # x is a numpy array with 2 elements.
    h1 = sigmoid(w1 * x[0] + w2 * x[1] + w3 * x[2] + b1)
    h2 = sigmoid(w4 * x[0] + w5 * x[1] + w6 * x[2] + b2)
    o1 = sigmoid(w7 * h1 + w8 * h2 + b3)
    return o1

def feedforward2(W1,W2,w7,w8,b1,b2,b3,x, data):
    # x is a numpy array with 2 elements.
    # x1
    x = x.reshape(data.shape[1],1)
    h1 = sigmoid(np.dot(W1,x) + b1)
    h2 = sigmoid(np.dot(W2,x) + b2)
    o1 = sigmoid(w7 * h1 + w8 * h2 + b3)
    return np.squeeze(o1)


# In[3]:


# X_xor=np.array([[0,1,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]

# X_xor=np.array([[0,0,1,1],[0,1,0,1]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1]


# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 

# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 


# In[4]:


# Weights
w7 = np.random.normal()
w8 = np.random.normal()

# Biases
b1 = np.random.normal()
b2 = np.random.normal()
b3 = np.random.normal()


# In[5]:


learn_rate_wg = 2.2
learn_rate_bias = 2.2
epochs = 1000 # number of times to loop through the entire dataset


# In[6]:


# if len(X_xor) == 1:
#     # W1 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
#     # W2 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
#     W1 = np.random.normal()
#     W2 = np.random.normal()
# else:
#     W1 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
#     W2 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
    
# for epoch in range(epochs):
#     for x, y_true in zip(data, all_y_trues):
#         # --- Do a feedforward (we'll need these values later)

#         # x1
#         x = x.reshape(data.shape[1],1)

#         # H1
# #         sum_h1_0 = w1 * x[0] + w2 * x[1] + w3*x[2] + b1
# #         h1_0 = sigmoid(sum_h1_0)

#         # H1 alternative
# #         W1 = np.random.normal(size=(1,3))
#         sum_h1 = np.dot(W1,x) + b1
#         h1 = sigmoid(sum_h1)

#         # H2
# #         sum_h2_0 = w4 * x[0] + w5 * x[1] + w6*x[2] + b2
# #         h2_0 = sigmoid(sum_h2_0)

#         # H2 alternative
# #         W2 = np.random.normal(size=(1,3))
#         sum_h2 = np.dot(W2,x) + b2
#         h2 = sigmoid(sum_h2)

#         sum_o1 = w7 * h1 + w8 * h2 + b3
#         o1 = sigmoid(sum_o1)
#         y_pred = o1

# #         d_L_d_ypred = -2 * (y_true - y_pred)
# #         m_true = y_true + 1
# #         print('M:',m_true)
# #         m = y_true + data.shape[0]
#         m = 4
# #             print('m:', m)
#         d_L_d_ypred = ((y_pred - y_true)/(y_pred*(1 - y_pred)))/(m)

#         # Neuron o1
#         d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
#         d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
#         d_ypred_d_b3 = deriv_sigmoid(sum_o1)/m

#         d_ypred_d_h1 = w7 * deriv_sigmoid(sum_o1)
#         d_ypred_d_h2 = w8 * deriv_sigmoid(sum_o1)

#         # Neuron h1
# #         d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
# #         d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
# #         d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
#         d_h1_d_b1 = deriv_sigmoid(sum_h1)/m
#         neuron_h1 = np.dot(x,deriv_sigmoid(sum_h1))

#         # Neuron h2
# #         d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
# #         d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
# #         d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
#         d_h2_d_b2 = deriv_sigmoid(sum_h2)/m
#         neuron_h2 = np.dot(x,deriv_sigmoid(sum_h2))

# #         for k in range(len(W1)):
# #             W1[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[k]
# #             W2[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[k]


#         # --- Update weights and biases
#         # Neuron h1
# #         w1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
# #         w2 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
# #         w3 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
#         b1 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
# #         for i in range(len(neuron_h1)):
# #         W1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1


#         # Neuron h2
# #         w4 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
# #         w5 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
# #         w6 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
#         b2 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
# #         for i in range(len(neuron_h2)):
# #         W2 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2

#         if len(neuron_h1) <= 1:
#             W1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1
#             W2 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2
#         else:
#             for i in range(len(neuron_h1)):
#                 W1[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[i]
#                 W2[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[i]


#         # Neuron o1
#         w7 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w7
#         w8 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w8
#         b3 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_b3

# #        --- Calculate total loss at the end of each epoch
# #         if epoch % 10 == 0:
# #             y_preds = np.apply_along_axis(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,x), 1, data)
# # #             y_preds = np.apply_along_axis(feedforward2(W1,W2,w7,w8,b1,b2,b3,x), 1, data)
# # #             loss = mse_loss(all_y_trues, y_preds)
# #             loss = cost(all_y_trues, y_preds)
# #             print("Epoch %d loss: %.5f" % (epoch, loss))


# In[7]:


def train_fit_2(data,all_y_trues,w7,w8,b1,b2,b3):

    np.random.seed(8)
    
    learn_rate_wg = 2.2
    learn_rate_bias = 2.2
    epochs = 1000 # number of times to loop through the entire dataset

    if len(X_xor) == 1:
        # W1 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
        # W2 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
        W1 = np.random.normal()
        W2 = np.random.normal()
    else:
        W1 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
        W2 = np.squeeze(np.random.normal(size=(1,data.shape[1])))

    for epoch in range(epochs):
        h1_sum = []
        h2_sum = []  
        for x, y_true in zip(data, all_y_trues):
            # --- Do a feedforward (we'll need these values later)

            # x1
            x = x.reshape(data.shape[1],1)

            # H1
    #         sum_h1_0 = w1 * x[0] + w2 * x[1] + w3*x[2] + b1
    #         h1_0 = sigmoid(sum_h1_0)

            # H1 alternative
    #         W1 = np.random.normal(size=(1,3))
            sum_h1 = np.dot(W1,x) + b1
            h1 = sigmoid(sum_h1)
            h1_sum.append(sum_h1)

            # H2
    #         sum_h2_0 = w4 * x[0] + w5 * x[1] + w6*x[2] + b2
    #         h2_0 = sigmoid(sum_h2_0)

            # H2 alternative
    #         W2 = np.random.normal(size=(1,3))
            sum_h2 = np.dot(W2,x) + b2
            h2 = sigmoid(sum_h2)
            h2_sum.append(sum_h2)

            sum_o1 = w7 * h1 + w8 * h2 + b3
            o1 = sigmoid(sum_o1)
            y_pred = o1
            
            hidden_layer = [h1_sum,h2_sum]

    #         d_L_d_ypred = -2 * (y_true - y_pred)
    #         m_true = y_true + 1
    #         print('M:',m_true)
    #         m = y_true + data.shape[0]
            m = 4
    #             print('m:', m)
            d_L_d_ypred = ((y_pred - y_true)/(y_pred*(1 - y_pred)))/(m)

            # Neuron o1
            d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
            d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
            d_ypred_d_b3 = deriv_sigmoid(sum_o1)/m

            d_ypred_d_h1 = w7 * deriv_sigmoid(sum_o1)
            d_ypred_d_h2 = w8 * deriv_sigmoid(sum_o1)

            # Neuron h1
    #         d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
    #         d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
    #         d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
            d_h1_d_b1 = deriv_sigmoid(sum_h1)/m
            neuron_h1 = np.dot(x,deriv_sigmoid(sum_h1))

            # Neuron h2
    #         d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
    #         d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
    #         d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
            d_h2_d_b2 = deriv_sigmoid(sum_h2)/m
            neuron_h2 = np.dot(x,deriv_sigmoid(sum_h2))

    #         for k in range(len(W1)):
    #             W1[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[k]
    #             W2[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[k]


            # --- Update weights and biases
            # Neuron h1
    #         w1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
    #         w2 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
    #         w3 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
            b1 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
    #         for i in range(len(neuron_h1)):
    #         W1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1


            # Neuron h2
    #         w4 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
    #         w5 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
    #         w6 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
            b2 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
    #         for i in range(len(neuron_h2)):
    #         W2 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2

            if len(neuron_h1) <= 1:
                W1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1
                W2 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2
            else:
                for i in range(len(neuron_h1)):
                    W1[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[i]
                    W2[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[i]


            # Neuron o1
            w7 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w7
            w8 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w8
            b3 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_b3

    #        --- Calculate total loss at the end of each epoch
    #         if epoch % 10 == 0:
    #             y_preds = np.apply_along_axis(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,x), 1, data)
    # #             y_preds = np.apply_along_axis(feedforward2(W1,W2,w7,w8,b1,b2,b3,x), 1, data)
    # #             loss = mse_loss(all_y_trues, y_preds)
    #             loss = cost(all_y_trues, y_preds)
    #             print("Epoch %d loss: %.5f" % (epoch, loss))
    
    return W1,W2,w7,w8,b1,b2,b3,hidden_layer


# In[8]:


# # # Make some predictions
# if len(X_xor) == 1:
#     test = np.array([0]) # 128 pounds, 63 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([0])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
#     # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[9]:


# def train_fit(data,all_y_trues,w7,w8,b1,b2,b3):
#     learn_rate_wg = 2.2
#     learn_rate_bias = 2.2
#     epochs = 1000 # number of times to loop through the entire dataset
    
#     W1 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
#     W2 = np.squeeze(np.random.normal(size=(1,data.shape[1])))
#     for epoch in range(epochs):
#         for x, y_true in zip(data, all_y_trues):
#             # --- Do a feedforward (we'll need these values later)

#             # x1
#             x = x.reshape(data.shape[1],1)

#             # H1
#     #         sum_h1_0 = w1 * x[0] + w2 * x[1] + w3*x[2] + b1
#     #         h1_0 = sigmoid(sum_h1_0)

#             # H1 alternative
#     #         W1 = np.random.normal(size=(1,3))
#             sum_h1 = np.dot(W1,x) + b1
#             h1 = sigmoid(sum_h1)

#             # H2
#     #         sum_h2_0 = w4 * x[0] + w5 * x[1] + w6*x[2] + b2
#     #         h2_0 = sigmoid(sum_h2_0)

#             # H2 alternative
#     #         W2 = np.random.normal(size=(1,3))
#             sum_h2 = np.dot(W2,x) + b2
#             h2 = sigmoid(sum_h2)

#             sum_o1 = w7 * h1 + w8 * h2 + b3
#             o1 = sigmoid(sum_o1)
#             y_pred = o1

#     #         d_L_d_ypred = -2 * (y_true - y_pred)
#             m_true = y_true + 1
#     #         print('M:',m_true)
#             m = y_true + data.shape[0]
#     #             print('m:', m)
#             d_L_d_ypred = ((y_pred - y_true)/(y_pred*(1 - y_pred)))/(m_true)

#             # Neuron o1
#             d_ypred_d_w7 = h1 * deriv_sigmoid(sum_o1)
#             d_ypred_d_w8 = h2 * deriv_sigmoid(sum_o1)
#             d_ypred_d_b3 = deriv_sigmoid(sum_o1)/m

#             d_ypred_d_h1 = w7 * deriv_sigmoid(sum_o1)
#             d_ypred_d_h2 = w8 * deriv_sigmoid(sum_o1)

#             # Neuron h1
#     #         d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
#     #         d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
#     #         d_h1_d_w3 = x[2] * deriv_sigmoid(sum_h1)
#             d_h1_d_b1 = deriv_sigmoid(sum_h1)/m
#             neuron_h1 = np.dot(x,deriv_sigmoid(sum_h1))

#             # Neuron h2
#     #         d_h2_d_w4 = x[0] * deriv_sigmoid(sum_h2)
#     #         d_h2_d_w5 = x[1] * deriv_sigmoid(sum_h2)
#     #         d_h2_d_w6 = x[2] * deriv_sigmoid(sum_h2)
#             d_h2_d_b2 = deriv_sigmoid(sum_h2)/m
#             neuron_h2 = np.dot(x,deriv_sigmoid(sum_h2))

#     #         for k in range(len(W1)):
#     #             W1[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[k]
#     #             W2[k] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[k]


#             # --- Update weights and biases
#             # Neuron h1
#     #         w1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
#     #         w2 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
#     #         w3 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w3
#             b1 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1
#     #         for i in range(len(neuron_h1)):
#     #             W1[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[i]


#             # Neuron h2
#     #         w4 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
#     #         w5 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w5
#     #         w6 -= learn_rate_wg* d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w6
#             b2 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2
#     #         for i in range(len(neuron_h2)):
#     #             W2[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[i]

#             if len(neuron_h1) <= 1:
#                 W1 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1
#                 W2 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2
#             else:
#                 for i in range(len(neuron_h1)):
#                     W1[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h1 * neuron_h1[i]
#                     W2[i] -= learn_rate_wg * d_L_d_ypred * d_ypred_d_h2 * neuron_h2[i]


#             # Neuron o1
#             w7 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w7
#             w8 -= learn_rate_wg * d_L_d_ypred * d_ypred_d_w8
#             b3 -= learn_rate_bias * d_L_d_ypred * d_ypred_d_b3
            

# #     #        --- Calculate total loss at the end of each epoch
# #             if epoch % 10 == 0:
# #                 y_preds = np.apply_along_axis(feedforward2(W1,W2,w7,w8,b1,b2,b3,x), 0, data)
# #     #             y_preds = np.apply_along_axis(feedforward2(W1,W2,w7,w8,b1,b2,b3,x), 1, data)
# #     #             loss = mse_loss(all_y_trues, y_preds)
# #                 loss = cost(all_y_trues, y_preds)
# #                 print("Epoch %d loss: %.5f" % (epoch, loss))
            
#     return W1,W2,w7,w8,b1,b2,b3

#     # print(d_h1_d_w1)
#     # print(d_h1_d_w2)
#     # print(d_h1_d_w3)
#     # print(W2)
#     # print(sum_h2_0)
#     # print(sum_h2)

#     # print(d_h1_d_w1)
#     # print(d_h1_d_w2)
#     # print(d_h1_d_w3)
#     # print(neuron_h1)

#     # print(d_h2_d_w4)
#     # print(d_h2_d_w5)
#     # print(d_h2_d_w6)
#     # print(neuron_h2)

#     # print(w1)
#     # print(w2)
#     # print(w3)
#     # print(W1)

#     # print(w4)
#     # print(w5)
#     # print(w6)
#     # print(W2)


# In[10]:


# X_xor=np.array([[0,1,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = 2

X_xor=np.array([[0,0,1,1],[0,1,0,1]])
all_y_trues=np.array([0,1,1,0])
data = X_xor.T
data.shape[0]
seed_num = data.shape[0] * data.shape[1]


# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0],[1,0,0,1],[1,1,0,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 

# X_xor=np.array([[0,0,1,1],[0,1,0,1],[1,0,1,0]])
# all_y_trues=np.array([0,1,1,0])
# data = X_xor.T
# data.shape[0]
# seed_num = data.shape[0] * data.shape[1] 


# In[11]:


np.random.seed(seed_num)
for _ in range(0,2):
    W1,W2,w7,w8,b1,b2,b3,hidden_layer = train_fit_2(data,all_y_trues,w7,w8,b1,b2,b3)
#     data = np.array(hidden_layer).reshape(4,2)
#     W1,W2,w7,w8,b1,b2,b3,hidden_layer = train_fit_2(data,all_y_trues,w7,w8,b1,b2,b3)
#     data = np.array(hidden_layer).reshape(4,2)
#     data = hl1

# # len(hidden_la)
hl1 = np.array(hidden_layer).reshape(4,2)
print('hl1:',hl1)

data2 = np.squeeze(hl1)
print('data:',data)
print('data2:',data2)


# for j in range(len(data2)):
#     for k in range(len(data2[0])):
#         if data2[j][k] >= 0.5:
#             data2[j][k] = 1
#         else:
#             data2[j][k] = 0
            
# print(data2)
# print(len(data2))

# np.random.seed(seed_num)
# test = np.array([1, 0]) # 128 pounds, 63 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
# test = np.array([0, 0])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
# test = np.array([0, 1])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
# test = np.array([1, 1])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))

# # Weights
# w7 = np.random.normal()
# w8 = np.random.normal()

# # Biases
# b1 = np.random.normal()
# b2 = np.random.normal()
# b3 = np.random.normal()

np.random.seed(seed_num)
for _ in range(0,2):
    W1,W2,w7,w8,b1,b2,b3,hl2 = train_fit_2(data,all_y_trues,w7,w8,b1,b2,b3)
    
# np.random.seed(seed_num)
# test = np.array([1, 0]) # 128 pounds, 63 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data2))
# test = np.array([0, 0])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data2))
# test = np.array([0, 1])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data2))
# test = np.array([1, 1])  # 155 pounds, 68 inches
# # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
# # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
# # print(network.feedforward2(W1,W2,test))
# print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data2))


# In[12]:


# # Make some predictions
if len(X_xor) == 1:
    test = np.array([0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
    # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx


# In[13]:


# # # Make some predictions
if len(X_xor) == 2:
#     np.random.seed(seed_num)
    test = np.array([1, 0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([0, 0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([0, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([1, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
    # print(network.feedforward2(W1,W2,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))

# # # # network.feedforward2(W1,W2,x)


# In[14]:


if len(X_xor) == 5:
    # np.random.seed(seed_num)
    # for _ in range(0,2):
    #     W1,W2,w7,w8,b1,b2,b3 = train_fit(data_5,all_y_trues,w7,w8,b1,b2,b3)

    # # Make some predictions
    # np.random.seed(7)
    np.random.seed(seed_num)
    test = np.array([1, 0, 1, 0, 0]) # 128 pounds, 63 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([0, 0, 0, 0, 0])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([0, 1, 1, 0, 1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))
    test = np.array([1, 1, 1, 1,1])  # 155 pounds, 68 inches
    # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
    print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test,data))


# In[15]:


# # # Make some predictions
# if len(X_xor) == 3:
#     np.random.seed(seed_num)
#     test = np.array([1, 0, 1]) # 128 pounds, 63 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([0, 0, 0])  # 155 pAounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([0, 0, 1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     test = np.array([1, 1, 1])  # 155 pounds, 68 inches
#     # print(feedforward(w1,w2,w3,w4,w5,w6,w7,w8,b1,b2,b3,test))
#     # print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     # print(network.feedforward2(W1,W2,test))
#     print(feedforward2(W1,W2,w7,w8,b1,b2,b3,test))
#     # # print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
#     # # print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - Mx

