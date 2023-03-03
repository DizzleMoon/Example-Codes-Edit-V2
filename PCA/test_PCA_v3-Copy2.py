#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=7)
import matplotlib.pyplot as plt
import math 
import pandas as pd
from numpy import linalg as LA
from sympy import * 


# In[2]:


# Scale function
def scale(vec_1):
    
    # Test
    v1_scale = np.square(np.linalg.norm(vec_1))
#     print(v1_scale)

    v1 = np.multiply(1/np.sqrt(v1_scale + 1e-15),vec_1)
#     print(v1)
    
    return list(v1)
    
    
def QR(span):
    
    # Setup arrays
    # Original Values
    vector = []
    # Normalized Values
    vec_norm = []

    # Size of array
    row,col = span.shape
    
    # First Column
    v1 = span[:,0]
    vec_1 = v1
    vector.append(vec_1)

    v1 = scale(vec_1)
    vec_norm.append(v1)
    
    # Subsequent Columns

    start = 1
    end = col

    for j in range(start,end):

        # Column
        projection = span[:,j]
        v = span[:,j]

        #Orthonormal vector
        for i in range(len(vec_norm)):
            #projection
            # dot
            proj_dot = np.dot(v,vec_norm[i])/(np.dot(vec_norm[i],vec_norm[i]) + 1e-15)
            proj = np.multiply(proj_dot,vec_norm[i])
            projection = projection - proj

        v_norm = scale(projection)

        vec_norm.append(v_norm)
        
    # Calculate R
    Q = vec_norm
    Q = np.array(Q)
    R = np.dot(Q,span)
    R = np.array(R)

    
    return Q.T,R

def eigenspace(span):
    
    eig_vec = np.eye(span.shape[0])
    X = span.copy()

    for _ in range(100):
        Q,R = QR(X)
        eig_vec = np.dot(eig_vec,Q)
        X = np.dot(R,Q)
        eig_val = np.diag(X)
        
    return eig_val,eig_vec, X


# In[3]:


def CoV(df):
    
    # Shape of dataframe
    row,col = df.shape
    print(row,col)
    
    # Determine indices
#     if row > col:
#         row = col
#         col = row
    
    # Initialize Covariance matrix
    CoVar = np.zeros((col,col))
    
    # List of means
    stats = []
    for i in range(col):
        stats.append(df.iloc[:,i].mean())

    # Solve covariance matrix        
    for k in range(col):
        for i in range(col):
            var = 0
            for j in range(row):
                var = var + ((df.iloc[j,k] - stats[k])*(df.iloc[j,i] - stats[i]))/(row-1)
            CoVar[k][i] = var
        
    return CoVar
    
    


# In[4]:


x = [2.5,0.5,2.2,1.9,3.1,2.3,2,1,1.5,1.1]
y = [2.4,0.7,2.9,2.2,3.0,2.7,1.6,1.1,1.6,0.9]

# x = [7,4,6,8,8,7,5,9,7,8]
# y = [4,1,3,6,5,2,3,5,4,2]
# z = [3,8,5,1,7,9,3,8,5,2]

# xyz = np.array([[7,4,3],[4,1,8],[6,3,5],[8,6,1],[8,5,7],[7,2,9],[5,3,3],[9,5,8],[7,4,5],[8,2,2]])
# X = pd.DataFrame(xyz)
# print(X)

f1 = [1,5,1,5,8]
f2 = [2,5,4,3,1]
f3 = [3,6,2,2,2]
f4 = [4,7,3,1,2]


# In[5]:


# Create dataframe.
df = pd.DataFrame([f1,f2,f3,f4])
df = df.transpose()
df_copy = df.copy()
# df.columns = ['x','y','z']
print(df)
print(df.shape)
row,col = df.shape
print(df.mean())
print(df.std())


# In[6]:


# # Adjust Data
# x_adjust = []
# y_adjust = []
# for i in range(len(x)):
#     x_adjust.append(df['x'][i] - df['x'].mean())
#     y_adjust.append(df['y'][i] - df['y'].mean())
    
# print(x_adjust)
# print(y_adjust)

# row_data_adjust = pd.DataFrame([x_adjust,y_adjust])
# print(row_data_adjust.transpose())


# In[7]:


# Covariance
# df_2 = (df - df.mean())/df.std()
df_2 = df - df.mean()
cov = CoV(df_2)
print(cov)


# In[8]:


# Eigenspace
eig_val,eig_vec, x_val = eigenspace(cov)
print(eig_val)
print(eig_vec)

# Check for orthonormal
orthonormality = np.linalg.norm(eig_vec[0])
print(orthonormality)

# Cosine Similarity
cos_sim = eig_vec[0].dot(eig_vec[1])
print(cos_sim)


# In[9]:


A = eig_vec[:,0:2]
print(A)


# In[10]:


# Final Step
# Select top two eigen values
A = eig_vec[:,0:2]
# A = pd.Series(A)
A = pd.DataFrame(A)
# A = A.transpose()
print(A)

# df_0 = (df - df.mean())/df.std()
df_0 = df - df.mean()
# df_0 = df_0.transpose()
print(df_0)
# df_1 = df_0.loc[0:3,:]
df_1 = df_0

# A*df_0.transpose()
Y = df_1.dot(A)
print(Y)


# In[11]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(df_copy)
Y = pca.transform(df_copy)
print(Y)


# In[12]:


# Recover data
x_0 = A.dot(Y.T)
print(x_0)
x_0 = x_0.T
x_mean = df.mean()
x_std = df.std()

for i in range(len(x_mean)):
    xhat = x_0[i] + x_mean[i]
    print(xhat)

# xhat_0 = x_0[0]*x_std[0] + x_mean[0]
# xhat_1 = x_0[1]*x_std[1] + x_mean[1]
# xhat_0 = x_0[0] + x_mean[0]
# xhat_1 = x_0[1] + x_mean[1]
# xhat_2 = x_0[2] + x_mean[2]
# print(xhat_0)
# print(xhat_1)
# print(xhat_2)
print(df_copy)

# mse = (((df_copy[0] - xhat_0)**2)/100).sum()
# print(mse)


# In[13]:


# df_0 = df - df.mean()
# print(df_0.loc[0])

# A = pd.DataFrame(A)
# A = A.transpose()
# print(A)

# # Y = A.loc[0].mul(df_0.loc[0])
# Y = 
# print(Y)

# # Y = np.matmul(A,df_0.transpose())
# # print(Y)

# # Y = A.T.dot((df-df.mean()).transpose)
# # print(Y)


# In[14]:


# # New data set
# row_data_adjust = pd.DataFrame([x_adjust,y_adjust])
# print(row_data_adjust)
# eig_vec = pd.DataFrame(eig_vec[0])
# print(eig_vec)
# eig_vec = eig_vec.transpose()
# row_feature_vec = eig_vec.dot(row_data_adjust)
# print(row_feature_vec)


# In[15]:


# rowdataadjust = row_feature_vec.transpose().dot(eig_vec)
# print(rowdataadjust)
# # x_adj = rowdataadjust.iloc[:,0] +  df['x'].mean()
# # print(x_adj)


# In[16]:


# X = np.array([x,y,z])
X = np.array([x,y])
X = X.T

Xmean = np.mean(X,0)

C = np.cov(X.T)


# In[17]:


# Eigenspace
eig_val,eig_vec, x_val = eigenspace(C)
print(eig_val)
print(eig_vec)

A = eig_vec[:,0:2]
print(A)


# In[18]:


print(A.T)
print((X-Xmean).T)
Y = np.matmul(A.T,(X-Xmean).T)
print(Y)


# In[19]:


df1 = pd.DataFrame([[0,1,1,2],[2,1,1,0]])
print(df1)
df2 = pd.DataFrame([[1,2],[2,3],[2,3],[4,1]])
print(df2)
df1.dot(df2)


# In[20]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(df_copy)
Y = pca.transform(df_copy)
print(Y)


# In[21]:


def PCA(df):
    # Create dataframe.
#     df = pd.DataFrame([f1,f2,f3,f4])
    df = df.transpose()
    df_copy = df.copy()
    # df.columns = ['x','y','z']
    print(df)
    print(df.shape)
    row,col = df.shape
    print(df.mean())
    print(df.std())
    
    
    # Covariance
    # df_2 = (df - df.mean())/df.std()
    df_2 = df - df.mean()
    cov = CoV(df_2)
    print(cov)
    
    # Eigenspace
    eig_val,eig_vec, x_val = eigenspace(cov)
    print(eig_val)
    print(eig_vec)
    
   
    # Final Step
    # Select top two eigen values
    A = eig_vec[:,0:2]
    # A = pd.Series(A)
    A = pd.DataFrame(A)
    # A = A.transpose()
    print(A)

    # df_0 = (df - df.mean())/df.std()
    df_0 = df - df.mean()
    # df_0 = df_0.transpose()
    print(df_0)
    # df_1 = df_0.loc[0:3,:]
    df_1 = df_0

    # A*df_0.transpose()
    Y = df_1.dot(A)
    print(Y)
    
#     print(A.T)
#     print((X-Xmean).T)
#     Y = np.matmul(A.T,(X-Xmean).T)
#     print(Y)
    
    return A,Y


# In[22]:


A,Y = PCA(df_copy)
print(A)
print(Y)


# In[23]:


from sklearn import decomposition
pca = decomposition.PCA(n_components=2)
pca.fit(df_copy)
Y = pca.transform(df_copy)
print(Y)


# In[24]:


# Recover data
x_0 = A.dot(Y.T)
print(x_0)
x_0 = x_0.T
x_mean = df.mean()
x_std = df.std()

xhat = []
for i in range(len(x_mean)):
    x_hat = x_0[i] + x_mean[i]
    xhat.append(x_hat)
    
xhat = pd.DataFrame(xhat)

# xhat_0 = x_0[0]*x_std[0] + x_mean[0]
# xhat_1 = x_0[1]*x_std[1] + x_mean[1]
# xhat_0 = x_0[0] + x_mean[0]
# xhat_1 = x_0[1] + x_mean[1]
# xhat_2 = x_0[2] + x_mean[2]
# print(xhat_0)
# print(xhat_1)
# print(xhat_2)
print(df_copy)

print(xhat.T)

print(df_copy.shape)

mse = (((df_copy - xhat.T)**2)/(df_copy.shape[0])).sum()
print(mse)


# In[ ]:




