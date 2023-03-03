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
from collections import *


# In[2]:


# Sentences
Sentence_1 = 'We went to the pizza place and you ate no pizza at all.'
Sentence_2 = 'I ate pizza with you yesterday at home.'
Sentence_3 = 'There’s no place like home'


# In[3]:


# Count
Sen_1 = Sentence_1.split()
# print(Sen_1)
Sen_2 = Sentence_2.split()
# print(Sen_2)
Sen_3 = Sentence_3.split()
# print(Sen_3)

# Dictionary
Sent_1 = Counter(Sen_1)
print(Sent_1)
Sent_2 = Counter(Sen_2)
# print(Sent_2)
Sent_3 = Counter(Sen_3)
# print(Sent_3)


print('\n')

count = []
for i in Sen_1:
    cnt = Sen_1.count(i)
    count.append(cnt)
    
print(count)
 
print(Sent_1['pizza'])


# In[4]:


# Combining vectors for sentence 1 & sentence 2
sentences_1_2 = Sen_1 + Sen_2
print(sentences_1_2)

# Dictionary
# Sent_1_2 = Counter(sentences_1_2)
# print(Sent_1_2)


# In[5]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_1 = Sent_1.copy()
print(Sent_1)

for key,value in Sent_2.items():
    akey = str(key)
#     print(key)
    if akey not in Sent_1.keys():
        print(key)
        doc_1[key] = 0
#         S1 = Sent_1[str(key)] + Sent_2[str(key)]
#         print(S1)
#     print(key)
#     print(value)

# print(doc_1)
print(sorted(doc_1.items()))

doc_1_val = []
for key,val in sorted(doc_1.items()):
    doc_1_val.append(val)
    
print(doc_1_val)


# In[6]:


doc_2 = Sent_2.copy()
print(doc_2)

for key,value in Sent_1.items():
    akey = str(key)
#     print(key)
    if akey not in Sent_2.keys():
        print(key)
        doc_2[key] = 0
#         S1 = Sent_1[str(key)] + Sent_2[str(key)]
#         print(S1)
#     print(key)
#     print(value)

print(sorted(doc_2.items()))

doc_2_val = []
for key,val in sorted(doc_2.items()):
    doc_2_val.append(val)
    
print(doc_2_val)


# In[7]:


doc_1_val = np.array(doc_1_val)
print(doc_1_val)
doc_2_val = np.array(doc_2_val)
print(doc_2_val)
top = doc_1_val.dot(doc_2_val)
bottom = np.linalg.norm(doc_1_val) * np.linalg.norm(doc_2_val)
cos1 = top/bottom
print(cos1)
print(doc_1_val.shape)
print(len(doc_1_val))


# In[8]:


# IDF
# Corpus size
Corpus = 2

# Document Frequency V1
doc = np.concatenate((doc_1_val, doc_2_val))
# print(doc)
# print(len(doc))
doc_2d = doc.reshape(2,int(len(doc)/2))
print(doc_2d[0][4])
tf_idf_1 = []
for i in range(len(doc_1_val)):
    df = 0
    for j in range(2):
#         print(doc_2d[i][j])
        df = df + doc_2d[j][i]
#         print(df)
        
    tfidf = np.log(2/df)
    tf_idf_1.append(tfidf)
print(tf_idf_1)  

tf_idf_2 = []
# Document Frequency V2
for i in zip(doc_1_val, doc_2_val):
    df = sum(i)
    tfidf = np.log(2/df)
    tf_idf_2.append(tfidf)
print(tf_idf_2)  


# In[ ]:




