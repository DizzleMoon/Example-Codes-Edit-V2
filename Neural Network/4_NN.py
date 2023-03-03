#!/usr/bin/env python
# coding: utf-8

# # Neural Networks from Scratch (in NumPy)

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


import numpy.random as npr


# ## Data
# 
# Use `keras` only to load the data.

# In[3]:


from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28 * 28)) / 255
x_test = x_test.reshape((-1, 28 * 28)) / 255
del mnist
type(x_train)


# In[4]:


figsize(25, 5)
for i in range(10):
  subplot(1, 10, i+1)
  imshow(x_train[i].reshape((28, 28)), cmap='gray')
  xlabel(y_train[i])


# ## Utilities

# In[ ]:


def g(z):
    """ sigmoid """
    return 1 / (1 + np.exp(-z))


# In[ ]:


def g_(z):
    """ derivative sigmoid """
    return g(z) * (1 - g(z))


# In[ ]:


def T(y, K):
  """ one hot encoding """
  y = np.array(y, dtype=np.uint8)
  one_hot = np.zeros((len(y), K))
  one_hot[np.arange(len(y)), y] = 1
  return one_hot


# In[ ]:


def reverse(l):
  return l[::-1]


# ## Building a model
# 
# Architecture:
# 1. Input layer: $28 \times 28 = 784$ nodes.
# 1. Hidden layer: $300$ nodes.
# 1. Output layer: $10$ nodes.

# ### Initializing weights

# In[ ]:


# inspired by:
# https://github.com/google/jax/blob/master/examples/mnist_classifier_fromscratch.py
def init_random_params(layer_sizes, rng=npr.RandomState(0)):
  return [rng.randn(nodes_in + 1, nodes_out) * np.sqrt(2 / (nodes_in + nodes_out))
          for nodes_in, nodes_out, in zip(layer_sizes[:-1], layer_sizes[1:])]


# In[10]:


weights = init_random_params([784, 500, 500, 10])
[x.shape for x in weights]


# ### Feedforward

# In[ ]:


def add_bias(x):
  """ x.shape: batch * feature size """
  bias = np.ones((len(x), 1))
  return np.hstack((bias, x))


# In[ ]:


def forward(weights, inputs):
  x = inputs
  for w in weights: 
    x = add_bias(x)
    x = x @ w
    x = g(x)

  return x


# In[13]:


preds = forward(weights, npr.random((1, 28 * 28)))
preds.shape, preds


# In[14]:


get_ipython().run_line_magic('timeit', 'forward(weights, npr.random((1, 28 * 28)))')


# ## Training

# ### Backprop

# In[ ]:


def backward(x, y, weights):
  """ single example """

  # Feed forward, save activations
  x = inputs
  activations = [inputs]
  for w in weights:
    x = add_bias(x)
    bla = x.copy()
    x = x @ w
    activations.append(x)
    x = g(x)

  predictions = x

  # Get deltas, error terms
  final_error = (predictions - y).T
  errors = [final_error]
  # don't do final layer, we just did
  # don't compute error for input!
  for i, act in enumerate(activations[1:-1]):
    error = weights[-(i+1)][1:, :] @ errors[i] * g_(act).T # ignore the first weight because we don't adjust the bias 
    errors.append(error)

  errors = reverse(errors)

  # Save the partial derrivatives
  grads = []

  for i in range(len(errors)):
    grad = (errors[i] @ add_bias(activations[i])) * (1 / len(y))
    grads.append(grad)

  return grads


# In[16]:


inputs = npr.random((1, 28 * 28))
y = npr.random((1, 10))
grads = backward(inputs, y, weights)
[a.shape for a in grads]


# In[17]:


get_ipython().run_line_magic('timeit', 'grads = backward(inputs, y, weights)')


# In[18]:


inputs = npr.random((128, 28 * 28))
y = npr.random((128, 10))
grads = backward(inputs, y, weights)
[a.shape for a in grads]


# ### Measuring performance

# In[ ]:


def stats(weights):
  m = len(y_test)
  preds = forward(weights, x_test)
  acc = (np.argmax(preds, axis=-1) == y_test).sum() / m * 100
  loss = np.sum(- np.log(preds[np.arange(m), y_test])) / m
  return {'acc': round(acc, 2), 'loss': round(loss, 2)}


# In[20]:


get_ipython().run_line_magic('timeit', 'stats(weights)')


# In[21]:


stats(weights)


# ### Training loop

# In[ ]:


weights = init_random_params([784, 500, 500, 10])


# In[23]:


lr = 0.001

for epoch in range(2):
  print('Starting epoch', epoch + 1)
  for i in range(len(x_train)):
    inputs = x_train[i][np.newaxis, :]
    labels = T([y_train[i]], K=10)
    grads = backward(inputs, labels, weights)
    for j in range(len(weights)):
      weights[j] -= lr * grads[j].T
    if i % 5000 == 0: print(stats(weights))


# In[ ]:




