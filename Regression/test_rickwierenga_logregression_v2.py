#!/usr/bin/env python
# coding: utf-8

# In[44]:


get_ipython().run_line_magic('pylab', 'inline')


# In[45]:


get_ipython().system('wget -O dataset.csv https://archive.ics.uci.edu/ml/machine')


# In[46]:


get_ipython().system('head -3 dataset.csv')


# In[47]:


import pandas as pd


# In[48]:


# Loading the data
df = pd.read_csv('breast-cancer-wisconsin.data', names=[
  "id number",
  "Clump Thickness",
  "Uniformity of Cell Size",
  "Uniformity of Cell Shape",
  "Marginal Adhesion",
  "Single Epithelial Cell Size",
  "Bare Nuclei",
  "Bland Chromatin",
  "Normal Nucleoli",
  "Mitoses",
  "Class"
])


# In[49]:


df.head()


# In[50]:


# Data cleaning
df = df.replace('?',np.NaN)
df.isna().sum()


# In[51]:


# To NumPy & feature selection
X = df[["Clump Thickness",
  "Uniformity of Cell Size",
  "Uniformity of Cell Shape",
  "Marginal Adhesion",
  "Single Epithelial Cell Size",
  "Bare Nuclei",
  "Bland Chromatin",
  "Normal Nucleoli",
  "Mitoses"
]].values.astype(np.float32)
X.shape

idx = np.where(np.isnan(X))
X[idx] = np.take(np.nanmedian(X, axis = 0), idx[1])

y = df['Class'].values
y.shape


# In[52]:


# Cleaning y
# we only run this cell once.
if y[0] == 2:
  y = np.array(y == 4, dtype=np.float32)

y.shape, y[:10]


# In[53]:


# Bias Factor
X = np.hstack((np.ones((len(X), 1)), X))
X[:10]


# In[54]:


# Stats
m, n = X.shape
K = 2
K, m, n


# In[55]:


# Training a model
theta = np.zeros(n)


# In[56]:


# The model
def g(z):
  """ sigmoid """
  return 1 / (1 + np.exp(-z))

def h(X, theta):
  return g(X @ theta)

preds = h(X, theta)
preds.shape, preds[:10]


# In[57]:


# Cost functions & Gradients
def J(preds, y):
  return 1/m * (-y @ np.log(preds) - (1 - y) @ np.log(1 - preds))

def compute_gradient(theta, X, y):
  preds = h(X, theta)
  gradient = 1/m * X.T @ (preds - y)
  return gradient

compute_gradient(theta, X, y)

preds = h(X, theta)
J(preds, y)


# In[58]:


# Training Loop

hist = {'loss': [], 'acc': []}
alpha = 0.1

for i in range(100):
  gradient = compute_gradient(theta, X, y)
  theta -= alpha * gradient

  # loss
  preds = h(X, theta)
  loss = J(preds, y)
  hist['loss'].append(loss)

  # acc
  c = 0
  for j in range(len(y)):
    if (h(X[j], theta) > .5) == y[j]:
      c += 1
  acc = c / len(y)
  hist['acc'].append(acc)

  # print stats
  if i % 10 == 0: print(loss, acc)


# In[59]:


# Training evaluation
figsize(15, 5)
subplot(1, 2, 1)
plot(hist['loss'])
xlabel('loss')
subplot(1, 2, 2)
plot(hist['acc'])
xlabel('accuracy')


# In[60]:


# Final Performance
hist['loss'][-1], hist['acc'][-1]


# In[61]:


# Accuracy
preds = h(X, theta) > 0.5
(preds == y).sum() / len(y)


# In[62]:


#  F1
def precision(preds, labels):
  tp = ((preds == 1) == (y == 1)).sum()
  fp = ((preds == 1) == (y == 0)).sum()
  return tp / (tp + fp)

precision(preds, y)


# In[63]:


def recall(preds, labels):
  tp = ((preds == 1) == (y == 1)).sum()
  fn = ((preds == 0) == (y == 1)).sum()
  return tp / (tp + fn)

recall(preds, y)


# In[64]:


def f1(preds, labels):
  return 2 * (precision(preds, labels) * recall(preds, labels)) / (precision(preds, labels) + recall(preds, labels))

f1(preds, y)


# In[65]:


# Optimizing model performance
recalls = []
for p in range(100):
  preds = (h(X, theta) > (p / 100))
  r = recall(preds, y)
  recalls.append(r)


# In[66]:


plot(recalls)


# In[67]:


# Graphics
figsize(20, 5)
x = np.arange(-10, 10, 0.1)
plot((0, 0), (0, 1), 'g-', label='x = 0')
plot((-10, 10), (0.5, 0.5), 'r-', label='y = 0.5')
plot(x, g(x), label='g')
legend()


# In[ ]:




