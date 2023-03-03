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


# tf_doc_1 = np.array([1,1,1,1,2,1,1,1,1,1,1,1])
# tf_doc_2 = np.array([1,1,1,1,1,1,1,1,0,0,0,0])

tf_doc_1 = np.array([1,1,1,1,2,1,1,1,1,1,1,1,0,0,0,0])
tf_doc_2 = np.array([0,0,0,0,1,0,0,1,1,0,1,0,1,1,1,1])

# tf_doc_1 = np.array([1,1,1,1,2,1,1,1,1,1,1,1,0,0,0])
# tf_doc_3 = np.array([0,0,0,0,0,0,0,1,1,0,0,0,1,1,1])


# In[3]:


top = tf_doc_1.dot(tf_doc_2)
bottom = np.linalg.norm(tf_doc_1) * np.linalg.norm(tf_doc_2)
cos1 = top/bottom
print(cos1)


# In[4]:


Caesar = np.array([11.4,8.3,2.3,11.2,0,0,0])
np.linalg.norm(Caesar)


# In[ ]:




