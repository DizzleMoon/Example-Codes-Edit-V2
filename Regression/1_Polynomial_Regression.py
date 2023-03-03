#!/usr/bin/env python
# coding: utf-8

# ## Polynomial Regression - ML from the Fundamentals (1)
# 
# * **Check out the corresponding blog post: https://rickwierenga.com/blog/ml-fundamentals/polynomial-regression.html**
# 
# * Full series: https://rickwierenga.com/blog/ml-fundamentals/

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# ## Data & normalization

# In[2]:


X = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).T
y = np.array([45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000])

m, n = X.shape


# In[3]:


plt.plot(X, y, 'rx')


# In[4]:


X = np.hstack((np.ones((m, 1)), X))


# In[5]:


X = np.hstack((X, (X[:, 1] ** 2).reshape((m, 1)), (X[:, 1] ** 3).reshape((m, 1)), (X[:, 1] ** 4).reshape((m, 1)))); X


# In[6]:


_, n = X.shape


# In[7]:


X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
np.mean(X[:, 1:], axis=0), np.std(X[:, 1:], axis=0)


# ## Hypothesis & predictions

# In[8]:


def h(X, theta):
  return X @ theta


# In[9]:


theta = np.random.random(n)


# In[10]:


predictions = h(X, theta)
predictions


# In[11]:


h(X, theta), y


# In[12]:


predictions = h(X, theta)
plt.plot(X[:, 1], predictions, label='predictions')
plt.plot(X[:, 1], y, 'rx', label='labels')
plt.legend()


# ## Loss

# In[13]:


def J(theta, X, y):
  return np.mean(np.square(h(X, theta) - y))


# In[14]:


J(theta, X, y)


# ## Training

# In[15]:


alpha = 0.01


# In[16]:


losses = []
for _ in range(5000):
  theta = theta - alpha * (1/m) * (X.T @ ((X @ theta) - y))
  losses.append(J(theta, X, y))


# In[17]:


predictions = h(X, theta)
plt.plot(X[:, 1], predictions, label='predictions')
plt.plot(X[:, 1], y, 'rx', label='labels')
plt.legend()


# In[18]:


plt.plot(losses)


# In[19]:


losses[-1]


# ## Normal equation

# In[20]:


# recompute theta
theta = np.linalg.pinv(X.T@X) @ X.T @ y


# In[21]:


predictions = h(X, theta)
plt.plot(X[:, 1], predictions, label='predictions')
plt.plot(X[:, 1], y, 'rx', label='labels')
plt.legend()


# ---
# 
# By [Rick Wierenga](https://twitter.com/rickwierenga/)
