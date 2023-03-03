#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# # 1. LOAD DATASET

# In[44]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)


# # 2. Summarize the Dataset

# In[45]:


#shape
print(dataset.shape)


# # 3. Peek The Data

# In[46]:


print(dataset.tail())


# # Statistical Summary

# In[47]:


print(dataset.describe())


# # Class Distribution
# Let’s now take a look at the number of instances (rows) that belong to each class. We can view this as an absolute count.
# 

# In[48]:


print(dataset.groupby('class').size())


# # 4. Data Visualization
# We now have a basic idea about the data. We need to extend that with some visualizations.
# 
# We are going to look at two types of plots:
# 
# <li>Univariate plots to better understand each attribute.
# <li>Multivariate plots to better understand the relationships between attributes.
# 

# # 4.1 Univariate Plots

# In[49]:


dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()


# In[50]:


dataset.hist()
plt.show()


# # 4.2 Multivariate Plots

# In[51]:


# scatter plot matrix
scatter_matrix(dataset)
plt.show()


# # 5. Evaluate Some Algorithms
# 
# Here is what we are going to cover in this step:
# 
# <li>Separate out a validation dataset.
# <li>Set-up the test harness to use 10-fold cross validation.
# <li>Build 5 different models to predict species from flower measurements
# <li>Select the best model.
# 

# # 5.1 Create a Validation Dataset
# 
# We need to know that the model we created is any good.
# 
# Later, we will use statistical methods to estimate the accuracy of the models that we create on unseen data. We also want a more concrete estimate of the accuracy of the best model on unseen data by evaluating it on actual unseen data.
# 
# That is, we are going to hold back some data that the algorithms will not get to see and we will use this data to get a second and independent idea of how accurate the best model might actually be.
# 
# We will split the loaded dataset into two, 80% of which we will use to train our models and 20% that we will hold back as a validation dataset.

# In[52]:


array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# We now have training data in the X_train and Y_train for preparing models and a X_validation and Y_validation sets that we can use later.

# # 5.2 Test Harness
# 
# We will use 10-fold cross validation to estimate accuracy.
# 
# This will split our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations of train-test splits.

# In[53]:


seed = 7
scoring = 'accuracy'


# We are using the metric of ‘accuracy‘ to evaluate models. This is a ratio of the number of correctly predicted instances in divided by the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). We will be using the scoring variable when we run build and evaluate each model next.

# # 5.3 Build Models
# 
# We don’t know which algorithms would be good on this problem or what configurations to use. We get an idea from the plots that some of the classes are partially linearly separable in some dimensions, so we are expecting generally good results.
# 
# Let’s evaluate 6 different algorithms:
# 
# <li>Logistic Regression (LR)
# <li>Linear Discriminant Analysis (LDA)
# <li>K-Nearest Neighbors (KNN).
# <li>Classification and Regression Trees (CART).
# <li>Gaussian Naive Bayes (NB).
# <li>Support Vector Machines (SVM).
#     
# This is a good mixture of simple linear (LR and LDA), nonlinear (KNN, CART, NB and SVM) algorithms. We reset the random number seed before each run to ensure that the evaluation of each algorithm is performed using exactly the same data splits. It ensures the results are directly comparable.
# 
# Let’s build and evaluate our five models:

# In[54]:


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# print(models)
# evaluate each model in turn
results = []
names = []
for name, model in models:
        kfold = model_selection.KFold(n_splits=10)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


# We can see that it looks like KNN has the largest estimated accuracy score.
# 
# We can also create a plot of the model evaluation results and compare the spread and the mean accuracy of each model. There is a population of accuracy measures for each algorithm because each algorithm was evaluated 10 times (10 fold cross validation).

# In[55]:


# Compare Algos
fig = plt.figure()
fig.suptitle('Algorthm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# # 6. Make Predictions
# The KNN algorithm was the most accurate model that we tested. Now we want to get an idea of the accuracy of the model on our validation set.
# 
# This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.
# 
# We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report.

# In[56]:


knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print("\t=============================================\n")
print(confusion_matrix(Y_validation, predictions))
print("\t=============================================\n")
print(classification_report(Y_validation, predictions))


# We can see that the accuracy is 0.9 or 90%. The confusion matrix provides an indication of the three errors made. Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and support showing excellent results (granted the validation dataset was small).
# 
# 

# # Another Aproach to do so

# In[57]:


from sklearn.datasets import load_iris
iris = load_iris()
print(iris.feature_names)
print(iris.target)
print(iris.target_names)
print(iris.target.shape)


# In[58]:


from sklearn.neighbors import KNeighborsClassifier


# In[59]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[60]:


X = iris.data
y = iris.target


# In[61]:


knn.fit(X, y)


# In[62]:


knn.predict(iris.data)


# # Using Different Model

# In[63]:


from sklearn.linear_model import LogisticRegression


# In[64]:


logrg = LogisticRegression()


# In[65]:


logrg.fit(X, y)


# In[66]:


logrg.predict(iris.data)


# # How to Choose the best Model
# We will use classication accuracy method
# 
# # 1. Train and Test on the same dataset 

# In[67]:


from sklearn import metrics
print(metrics.accuracy_score(y, knn.predict(X)))
print(metrics.accuracy_score(y, logrg.predict(X)))


# # 2. Train/Test split

# In[68]:


from sklearn.model_selection import train_test_split


# In[69]:


print(X.shape)
print(y.shape)


# In[70]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=4)
# Random state is given any random integer value so that each time we train and test our model the values remain same.


# In[71]:


print(X_train.shape)
print(X_test.shape)


# In[72]:


print(y_train.shape)
print(y_test.shape)


# # Using K-Neighbours Classifier

# In[73]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)


# In[74]:


y_predict = knn.predict(X_test)
print(y_predict)


# In[75]:


print(metrics.accuracy_score(y_test, y_predict))


# # Using Logistic Regression

# In[76]:


logrg = LogisticRegression()
logrg.fit(X_train, y_train)


# In[77]:


y_predict = logrg.predict(X_test)
print(y_predict)


# In[78]:


print(metrics.accuracy_score(y_test, y_predict))
print(metrics.confusion_matrix(y_test, y_predict))


# # As the accuracy score of KNN model with n_neighbours=5 is more then the Logistic Regression Model.
# # Hence We will use KNN model for this problem
