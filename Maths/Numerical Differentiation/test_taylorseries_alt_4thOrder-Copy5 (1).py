#!/usr/bin/env python
# coding: utf-8

# In[53]:


import sympy as sym
import numpy as num
import matplotlib.pyplot as pyplot

import sympy as sy
import numpy as np
from sympy.functions import sin,cos,exp
import matplotlib.pyplot as plt
from sympy import* 
import math
plt.style.use("ggplot")

pyplot.style.use("ggplot")


# In[54]:


x = 0
h = 0.001
n = 4
term = 3
# f = lambda x: np.exp(x)


# In[55]:


# Create list of zeros
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j

# Middle function
mat_list[0,m] = 1
print(mat_list)


# In[56]:


for i in range(0,4):
    print(1/math.factorial(i+1)*(-1)**(i+1))


# In[57]:


# Create list of zeros
q = 0
mat_list = np.matrix(np.zeros((n,n)))
print(mat_list)
# m = floor(n/2)
m = 2
print(m)
for j in range(m): 
    for i in range(1,n+1):
        mat_list[i-1,3*j] = ((m**(i))/math.factorial(i)) 
        mat_list[i-1,3*j-2]=(1/math.factorial(i+1))
        if j > (m/2 - 1):
            mat_list[i-1,3*j] = ((m**(i))/math.factorial(i))*(-1)**(i+1)
            mat_list[i-1,3*j-1]=(1/math.factorial(i+1)) * (-1)**(i+1)            
print(mat_list)


# In[58]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
b[0][term-1] = math.factorial(term-1)
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[59]:


# Create list of zeros
n = 7
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m+1):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j/math.factorial(j)

# Middle function
mat_list[0,m] = 1
print(mat_list.T)


# In[60]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][4] = math.factorial(3)
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[61]:


# f_0 = 'x**3 + x - 1'
# f_1 = str(f_0)
f = lambda x: np.sin(x)


# In[62]:


i = 0
h = 0.001
# a = terms
func = np.matrix([f(i+3*h),f(i+2*h),f(i+h),f(i),f(i-h),f(i-2*h),f(i-3*h)])/(h**4)
print(func)


# In[63]:


for i in range(5):
    deriv = np.matmul(func,c)/math.factorial(i)
    print(deriv)


# In[64]:


# Create list of zeros
n = 4
mat_list = np.zeros((n,int(n/2)))
mat_list_2 = np.zeros((n,int(n/2)))
m = int(n/2)
print(m)
# First half
print(n)

# First Half
for i in reversed(range(m)):
    for j in range(n):
    #     mat_list[j,0] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
print(mat_list)

# Second Half
for i in range(m):
    for j in range(n):
    #     mat_list[j,0] = ((m-i)**j)/math.factorial(j)
        mat_list_2[j,i] = ((m-i)**j)*(-1)**j/math.factorial(j)
# mat_list_2[:,[0,1]] = mat_list_2[:,[1,0]]

print(mat_list_2)

# Concatenate 
mat_list_final = np.hstack((mat_list,np.flip(mat_list_2,1)))
print(mat_list_final)


# In[65]:


# Index
print(term)
b = np.zeros((1,4))
# print(b)
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][1] = 12
print(b)
# Weights
c = np.linalg.pinv(np.matrix(mat_list_final))*b.T
print(c)


# In[66]:


# Create list of zeros
n = 5
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m+1):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j/math.factorial(j)

# Middle function
mat_list[0,m] = 1
print(mat_list)


# In[67]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][2] = 12
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[68]:


# Create list of zeros
n = 6
mat_list = np.zeros((n,int(n/2)))
mat_list_2 = np.zeros((n,int(n/2)))
m = int(n/2)
print(m)
# First half
print(n)

# First Half
for i in reversed(range(m)):
    for j in range(n):
    #     mat_list[j,0] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
print(mat_list)

# Second Half
for i in range(m):
    for j in range(n-1,-1,-1):
    #     mat_list[j,0] = ((m-i)**j)/math.factorial(j)
        mat_list_2[j,i] = ((m-i)**j)*(-1)**j/math.factorial(j)
# mat_col = [0:2]
# mat_list_2[:,[0,1,2]] = mat_list_2[:,[2,1,0]]

# print(mat_list_2)
# print("\n")
# matt=np.flip(mat_list_2,1)
# print(matt)
# print(mat_list_2)

# Concatenate 
mat_list_final = np.hstack((mat_list,np.flip(mat_list_2,1)))
print(mat_list_final)


# In[69]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][3] = 8
print(b)
# Weights
c = np.linalg.pinv(np.matrix(mat_list_final))*b.T
print(c)


# In[70]:


for i in range(3,-1,-1):
    print(i)


# In[71]:


# Create list of zeros
n = 7
mat_list = np.matrix(np.zeros((n,n)))
m = floor(n/2)
print(m)
# First half
print(n)
for i in range(m+1):
#     print(i)
    for j in range(n):
        mat_list[j,i] = ((m-i)**j)/math.factorial(j)
        mat_list[j,i + m] = ((i)**j)*(-1)**j/math.factorial(j)

# Middle function
mat_list[0,m] = 1
print(mat_list.T)


# In[72]:


# Index
print(term)
b = np.zeros((1,n))
# b[0][term-1] = math.factorial(term-1)/(h**(term-1))
# b[0][term-1] = math.factorial(term-1)
b[0][4] = math.factorial(3)
print(b)
# Weights
c = np.linalg.pinv(mat_list)*b.T
print(c)


# In[73]:


h = 0.0001
i= 0.01
fac = 13
term = 13
f = lambda x: np.sin(x)

func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
print(func)

sign = 2
rem = term%4
if rem == 3:
    sign = 1  
print(sign)
deriv = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (sign)

# deriv = deriv*(h**(term-1))

print(deriv)


# In[74]:


h = 0.0037
# h = 0.0009000000000000002 * 6
i= 0.01
fac = 13
term = 12
f = lambda x: np.cos(x)

func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
print(func)

sign = 1
rem = (term)%4
if rem == 0 or rem == 3:
    sign = 2    
print(sign)

deriv = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (sign)

# deriv = deriv*(h**(term-1))

print(deriv)


# In[75]:


term=16
(term)%4


# In[76]:


from sympy import*
var( 'x' )
formula = sin(x)
series(formula, x, 0, 18).removeO()


# In[77]:


tol = 1e-3
h = 0.003
term = 16
i = 0.01
func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
deriv_old = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (2)
f = lambda x: np.cos(x)
while tol > 1e-15:
#     h = 0.01/j

    func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
#     print(func)

    sign = 1
    rem = (term)%4
    if rem == 0:
        sign = 2    
#     print(sign)

    deriv = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (sign)
    print(deriv)
    
    tol = abs(deriv_old - deriv)
    print(float(tol))
    deriv_old = deriv
    h += 0.0001*(6+1)

print(h)


# In[78]:


tol = 1e-3
h = 0.0001
term = 15
i = 0.01
func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
deriv_old = (np.matmul(func,c)/6/math.factorial(term)) * (-1) ** (2)
f = lambda x: np.sin(x)
while tol > 1e-15:
#     h = 0.01/j

    func = np.matrix([f(i-3*h),f(i-2*h),f(i-h),f(i),f(i+h),f(i+2*h),f(i+3*h)])/(h**4)
#     print(func)

    sign = 1
    rem = (term)%4
    if rem == 0:
        sign = 2    
#     print(sign)

    deriv = (np.matmul(func,c)/math.factorial(term)) * (-1) ** (2)
    print(deriv)
    
    tol = abs(deriv_old - deriv)
    print(float(tol))
    deriv_old = deriv
    h += 0.00001*(6+1)

print(h)


# In[ ]:




