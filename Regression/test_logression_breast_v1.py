#!/usr/bin/env python
# coding: utf-8

# In[961]:


import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import datasets, linear_model


# In[962]:


class LogisticRegression(object):
    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = np.ones(X.shape[1])

        for _ in range(self.n_iter):
            output = X.dot(self.w)
            errors = y - self.sigmoid(output)
            self.w += self.eta * errors.T.dot(X)
            # print(sum(errors**2) / 2.0)
        return self

    def predict(self, X):
        output = np.insert(X, 0, 1, axis=1).dot(self.w)
        return (self.sigmoid(output) + .5) / 1

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


# In[963]:


class LogisticRegressionOVR(object):
    """One vs Rest"""

    def __init__(self, eta=0.001, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.w = []

        for i in np.unique(y):
            y_copy = [1 if c == i else 0 for c in y]
            w = np.ones(X.shape[1])

            # print('training ', i)
            # counter = 0

            for _ in range(self.n_iter):
                output = X.dot(w)
                errors = y_copy - self.sigmoid(output)
                w += self.eta * errors.T.dot(X)
                
                # counter += 1
                # if counter // 10 == 0:
                #     print(sum(errors**2) / 2.0)
            self.w.append((w, i))

        return self


    def predictOne(self, x):
        return max((x.dot(w), c) for w, c in self.w)[1]

    def predict(self, X):
        return np.array([self.predictOne(i) for i in np.insert(X, 0, 1, axis=1)])

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))
    
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


# In[964]:


def main():
    iris = datasets.load_iris()
    # X = iris.data[:100, :2]
    # y = iris.target[:100]
    X = iris.data[:, :2]
    y = iris.target
    logi = LogisticRegression()
    logi.fit(X, y)
    y_pred = logi.predict(X)
    print(y_pred)
    print("out of a total %d points : %d" % (X.shape[0],(y != y_pred).sum()))


# In[965]:


# breast = datasets.load_breast_cancer()
# X = breast.data
# y = breast.target
# # y = y.reshape(len(y),1)
# logi = LogisticRegressionOVR(1000).fit(X, y)
# # print(logi.w)
# # y.shape


# In[987]:


np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target
logi = LogisticRegressionOVR(n_iter=1000).fit(X, y)
print(logi.w)
print(logi.predict(X_train))


# In[967]:


# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# logi = LogisticRegression(n_iter=1000).fit(X, y)
# print(logi.w)


# In[968]:


from sklearn.model_selection import train_test_split

# iris = datasets.load_iris()
X_train, X_temp, y_train, y_temp = train_test_split(breasts.data, breasts.target, test_size=.4)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=.5)


# In[969]:


# X = iris.data[:,:2]
# y = iris.target
# logi = LogisticRegression().fit(X_train,y_train)
logi.score(X_train,y_train)
# print(wg)
# logi.score(X_train,y_train)


# In[970]:



def fit(X, y):
    X = np.insert(X, 0, 1, axis=1)
    w = np.ones(X.shape[1])
    m = X.shape[0]
    eta = 0.001

    for _ in range(50):
        output = X.dot(w)
        errors = y - sigmoid(output)
        w += eta / m * errors.dot(X)
    return w

# def fit(X, y):
#     X = np.insert(X, 0, 1, axis=1)
#     w = []

#     for i in np.unique(y):
#         y_copy = [1 if c == i else 0 for c in y]
#         w1 = np.ones(X.shape[1])
#         eta = 0.001

#         # print('training ', i)
#         # counter = 0

#         for _ in range(50):
#             output = X.dot(w1)
#             errors = y_copy - sigmoid(output)
#             w1 += eta * errors.T.dot(X)

#             # counter += 1
#             # if counter // 10 == 0:
#             #     print(sum(errors**2) / 2.0)
#         w.append(w1)

#     return w

def predict(X,w):
    output = np.insert(X, 0, 1, axis=1).dot(w.T)
    return (np.floor(sigmoid(output) + .5)).astype(int)

def score(X, y,w):
    return sum(predict(X,w) == y) / len(y)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[971]:


# def fit(X, y):
#     X = np.insert(X, 0, 1, axis=1)
#     w = []

#     for i in np.unique(y):
#         y_copy = [1 if c == i else 0 for c in y]
#         w1 = np.ones(X.shape[1])
#         eta = 0.001

#         # print('training ', i)
#         # counter = 0

#         for _ in range(50):
#             output = X.dot(w1)
#             errors = y_copy - sigmoid(output)
#             w1 += eta * errors.T.dot(X)

#             # counter += 1
#             # if counter // 10 == 0:
#             #     print(sum(errors**2) / 2.0)
#         w.append((w1, i))

#     return w


# def predictOne(x,w):
#     return max((x.dot(w), c) for w, c in w)[1]

# def predict(X,w):
#     return np.array([predictOne(i,w) for i in np.insert(X, 0, 1, axis=1)])

# def sigmoid(x):
#     return 1.0 / (1 + np.exp(-x))

# def score(X, y, w):
#     return sum(predict(X,w) == y) / len(y)


# In[984]:


np.set_printoptions(precision=3)
breasts = datasets.load_breast_cancer()
X = breasts.data
y = breasts.target

X_train, X_temp, y_train, y_temp = train_test_split(breasts.data, breasts.target, test_size=.4)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=.5)

w = fit(X,y)
# print(X.shape)
# print(np.array(w).shape)
# w1 = predict(X,w)
w2 = score(X_train,y_train,w)
w2
w


# In[983]:


len(X_train[0])


# In[980]:


true_positives = false_positives = true_negatives = false_negatives = 0

for x_i, y_i in zip(X_train, y_train):
    print(x_i)
    print(len(x_i))
    prediction = sigmoid(np.dot(w[1:], x_i))

    if y_i == 1 and prediction >= 0.5:  # TP: paid and we predict paid
        true_positives += 1
    elif y_i == 1:                      # FN: paid and we predict unpaid
        false_negatives += 1
    elif prediction >= 0.5:             # FP: unpaid and we predict paid
        false_positives += 1
    else:                               # TN: unpaid and we predict unpaid
        true_negatives += 1

precision = true_positives / (true_positives + false_positives)
recall = true_positives / (true_positives + false_negatives)

true_positives


# In[974]:


print(precision)
print(recall)


# In[ ]:




