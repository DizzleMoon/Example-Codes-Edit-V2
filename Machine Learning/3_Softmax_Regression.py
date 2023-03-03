#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


import pandas as pd


# In[3]:


# !wget -O dataset.csv https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data


# In[4]:


# !head -3 dataset.csv


# In[5]:


df = pd.read_csv('iris.csv', names=[
  "sepal length in cm",
  "sepal width in cm",
  "petal length in cm",
  "petal width in cm",
  "class"
])


# In[6]:


df.head()


# In[7]:


X = df[["sepal length in cm",
  "sepal width in cm",
  "petal length in cm",
  "petal width in cm"
]].values.astype(np.float32)
X.shape


# In[8]:


y = pd.factorize(df['class'])[0]
y.shape


# In[9]:


X = np.hstack((np.ones((len(X), 1)), X))


# In[10]:


m, n = X.shape
K = 3
K, m, n


# In[11]:


X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)


# In[12]:


np.random.seed(0)
theta = np.random.random((n, K))


# In[13]:


def softmax(z):
    z -= np.max(z)
    return np.exp(z) / np.sum(np.exp(z))


# In[14]:


def h(X, theta):
  return softmax(X @ theta)


# In[15]:


def J(preds, y):
  return np.sum(- np.log(preds[np.arange(m), y]))


# In[16]:


def T(y, K):
  """ one hot encoding """
  one_hot = np.zeros((len(y), K))
  one_hot[np.arange(len(y)), y] = 1
  return one_hot


# In[17]:


def compute_gradient(theta, X, y):
  preds = h(X, theta)
  gradient = 1/m * X.T @ (preds - T(y, K))
  return gradient


# In[18]:


hist = {'loss': [], 'acc': []}
alpha = 1e-3

for i in range(1500):
  gradient = compute_gradient(theta, X, y)
  theta -= alpha * gradient

  # loss
  preds = h(X, theta)
  loss = J(preds, y)
  hist['loss'].append(loss)

  # acc
  c = 0
  for j in range(len(y)):
    if np.argmax(h(X[j], theta)) == y[j]:
      c += 1
  acc = c / len(y)
  hist['acc'].append(acc)

  # print stats
  if i % 200 == 0: print('{:.2f} {:.2f}%'.format(loss, acc * 100))


# In[19]:


figsize(10, 10)
subplot(2, 1, 1)
plot(hist['loss'])
xlabel('loss')
subplot(2, 1, 2)
plot(hist['acc'])
xlabel('accuracy')


# ## Graphics

# The rest of the notebook is not directly related to the concept. This section just shows how I generate graphics for the blog post.

# In[20]:


# error for $log 0$
x = np.linspace(0, 1, 100)
print(x.shape)
y = -np.log(x)
plot(x, y)
ylabel('-log x')
show()


# ---
# 
# By [Rick Wierenga](https://twitter.com/rickwierenga/)
