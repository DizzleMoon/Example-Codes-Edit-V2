#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import numpy as np
import random

from sklearn import datasets, linear_model


class LogisticRegression(object):
    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = np.ones(X.shape[1])

        for _ in range(self.n_iter):
            output = X.dot(self.w_)
            errors = y - self.sigmoid(output)
            self.w_ += self.eta * errors.T.dot(X)
            # print(sum(errors**2) / 2.0)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w_)
        return (self.sigmoid(output) + .5) // 1

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


class LogisticRegressionOVR(object):
    """One vs Rest"""

    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w_ = []

        for i in np.unique(y):
            y_copy = [1 if c == i else 0 for c in y]
#             w = np.ones(X.shape[1])
            w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

            # print('training ', i)
            # counter = 0

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self.sigmoid(output)
                w += self.eta * errors.T.dot(X)
                
                # counter += 1
                # if counter // 10 == 0:
                #     print(sum(errors**2) / 2.0)
            self.w_.append((w, i))

        return self


    def predictOne(self, x):
        return max((x.dot(w), c) for w, c in self.w_)[1]

    def predict(self, X):
        return np.array([self.predictOne(i) for i in np.insert(X, 0, 1, axis=1)])
    
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def softmax(self, x):
        """ applies softmax to an input x"""
        e_x = np.exp(x)
        return e_x / e_x.sum()
    
    def _net_input(self, X):
        """Compute the linear net input."""
        return (X.dot(self.w_) + self.b_).flatten()

    def _sigmoid_activation(self, z):
        """Compute the output of the logistic sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def _forward(self, X):
        z = self._net_input(X)
        a = self._sigmoid_activation(z)
        return a
    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        Class 1 probability : float
        """
        return self._forward(X)


def main():
    iris = datasets.load_iris()
    # X = iris.data[:100, :2]
    # y = iris.target[:100]
    X = iris.data[:, :2]
    y = iris.target
    logi = LogisticRegressionOVR()
    logi.fit(X, y)
    y_pred = logi.predict(X)
    print(y_pred)
    print("out of a total %d points : %d" % (X.shape[0],(y != y_pred).sum()))


# if __name__ == '__main__': main()

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh

# logreg = linear_model.LogisticRegression(C=1e5)
logreg = LogisticRegressionOVR(n_iter=1000)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


# In[12]:


logreg.w_


# In[13]:


logreg.sigmoid(Y)


# In[14]:


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp =     train_test_split(iris.data, iris.target, test_size=.4)
X_validation, X_test, y_validation, y_test =     train_test_split(X_temp, y_temp, test_size=.5)

logi = LogisticRegressionOVR(n_iter=1000).fit(X_train, y_train)
# logi = LogisticRegressionOVR(n_iter=1000).fit(X_validation, y_validation)

print(logi.score(X_train, y_train))
print(logi.score(X_validation, y_validation))


# In[15]:


logi.softmax(y_validation)


# In[16]:


logi.sigmoid(y_train)


# In[19]:


logi.w_


# In[ ]:


logi._

