#!/usr/bin/env python
# coding: utf-8

# In[3]:


class LinearRegressionSGD(object):
    def __init__(self, eta=0.1, n_iter=50, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
#         self.w = np.ones(X.shape[1])
        self.w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

#         for _ in range(self.n_iter):
#             if self.shuffle:
#                 X, y = self._shuffle(X, y)
#             # Your code here

        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            for x, target in zip(X, y):
                output = x.dot(self.w)
                error = target - output
                self.w += self.eta * error * x
                
        return self

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]    
    
    def predict(self, X):
        return np.insert(X, 0, 1, axis=1).dot(self.w)

    def score(self, X, y):
        return 1 - sum((self.predict(X) - y)**2) / sum((y - np.mean(y))**2)


# In[4]:


import numpy as np
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 1, 2, 3])
regr = LinearRegressionSGD().fit(X, y)
print(regr.w)
print(regr.score(X, y))
# print(regr.predict(X_test))


# In[5]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

diabetes = datasets.load_diabetes()
# Use only one feature
X = diabetes.data[:, np.newaxis, 2]
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

scaler_x = StandardScaler()
X_train = scaler_x.fit_transform(X_train)
X_test = scaler_x.transform(X_test)

regr = LinearRegressionSGD(eta=.1, n_iter=1500)
regr.fit(X_train, y_train)

plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


# In[6]:


X_train


# In[ ]:




