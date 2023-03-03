#!/usr/bin/env python
# coding: utf-8

# In[67]:


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


# In[68]:


logreg.w_


# In[69]:


logreg.sigmoid(Y)


# In[83]:


from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp =     train_test_split(iris.data, iris.target, test_size=.4)
X_validation, X_test, y_validation, y_test =     train_test_split(X_temp, y_temp, test_size=.5)

logi = LogisticRegressionOVR(n_iter=1000).fit(X_train, y_train)
# logi = LogisticRegressionOVR(n_iter=1000).fit(X_validation, y_validation)

print(logi.score(X_train, y_train))
print(logi.score(X_validation, y_validation))

print(logi.score(iris.data,iris.target))


# In[71]:


logi.softmax(y_validation)


# In[72]:


logi.sigmoid(y_train)


# In[73]:


logi.w_
log_point = logi.w_
log_point = log_point[0][0]
log_point_2 = log_point
log_point_2
# log_point_2
len(X_test)
y_test.shape
# _test.reshape(5,6)
# log_point_2.shape


# In[78]:


# # z = test @ logi.w
# abd = np.array(aw1[:3][0])
# abd
# test[0]
# logi.w
# z = np.matmul(Y,abd)
z = np.dot(X_test.reshape(24,5),log_point_2.reshape(5,1))
# # type(abd)
log_point_2.reshape(5,1)
X_test
logi.w_
z


# In[79]:


def sigmoid(x):
      return 1.0 / (1 + np.exp(-x))
  
h = sigmoid(z)
h


# In[81]:


iris.data


# In[ ]:




