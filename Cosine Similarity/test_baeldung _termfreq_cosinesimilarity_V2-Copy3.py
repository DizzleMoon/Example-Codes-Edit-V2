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
# Sentence_1 = 'We went to the pizza place and you ate no pizza at all.'
# Sentence_2 = 'I ate pizza with you yesterday at home.'
# Sentence_3 = 'There’s no place like home'

# Sentence_1 = 'The sky is blue'
# Sentence_2 = 'The sky is not blue'

Sentence_1 = 'The sun is the largest celestial body in the solar system'
Sentence_2 = 'The solar system consists of the sun and eight revolving planets'
Sentence_3 = 'Ra was the Egyptian Sun God'
Sentence_4 = 'The Pyramids were the pinnacle of Egyptian architecture'
Sentence_5 = 'The quick brown fox jumps over the lazy dog'


# In[3]:


# Count
Sen_1 = Sentence_1.split()
# print(Sen_1)
Sen_2 = Sentence_2.split()
# print(Sen_2)
Sen_3 = Sentence_3.split()
# print(Sen_3)
Sen_4 = Sentence_4.split()
# print(Sen_3)
Sen_5 = Sentence_5.split()
# print(Sen_3)

# Dictionary
Sent_1 = Counter(Sen_1)
print(Sent_1)
Sent_2 = Counter(Sen_2)
# print(Sent_2)
Sent_3 = Counter(Sen_3)
print(Sent_3)
Sent_4 = Counter(Sen_4)
print(Sent_3)
Sent_5 = Counter(Sen_5)
print(Sent_5)


print('\n')

count = []
for i in Sen_1:
    cnt = Sen_1.count(i)
    count.append(cnt)
    
print(count)
 
print(Sent_1['pizza'])


# In[4]:


# Combining vectors for sentence 1 & sentence 2
# sentences_1_2 = Sen_1 + Sen_2
# print(sentences_1_2)

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
    if akey not in zip(Sent_1.keys(),Sent_3.keys(),Sent_4.keys()):
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
    if akey not in zip(Sent_2.keys(),Sent_3.keys(),Sent_4.keys(),Sent_5.keys()):
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


doc_3 = Sent_3.copy()
print(doc_3)

for key,value in Sent_3.items():
    akey = str(key)
#     print(key)
    if akey not in zip(Sent_1.keys(),Sent_2.keys(),Sent_3.keys(),Sent_4.keys(),Sent_5.keys()):
        print(key)
        doc_3[key] = 0
#         S1 = Sent_1[str(key)] + Sent_2[str(key)]
#         print(S1)
#     print(key)
#     print(value)

print(sorted(doc_3.items()))

doc_3_val = []
for key,val in sorted(doc_3.items()):
    doc_3_val.append(val)
    
print(doc_3_val)


# In[8]:


doc_4 = Sent_4.copy()
print(doc_4)

for key,value in Sent_4.items():
    akey = str(key)
#     print(key)
    if akey not in zip(Sent_1.keys(),Sent_2.keys(),Sent_3.keys(),Sent_5.keys()):
        print(key)
        doc_4[key] = 0
#         S1 = Sent_1[str(key)] + Sent_2[str(key)]
#         print(S1)
#     print(key)
#     print(value)

print(sorted(doc_4.items()))

doc_4_val = []
for key,val in sorted(doc_4.items()):
    doc_4_val.append(val)
    
print(doc_4_val)


# In[9]:


doc_5 = Sent_5.copy()
print(doc_5)

for key,value in Sent_5.items():
    akey = str(key)
#     print(key)
    if akey not in zip(Sent_1.keys(),Sent_2.keys(),Sent_3.keys(),Sent_4.keys()):
        print(key)
        doc_5[key] = 0
#         S1 = Sent_1[str(key)] + Sent_2[str(key)]
#         print(S1)
#     print(key)
#     print(value)

print(sorted(doc_5.items()))

doc_5_val = []
for key,val in sorted(doc_5.items()):
    doc_5_val.append(val)
    
print(doc_5_val)


# In[10]:


doc_1_val = np.array(doc_1_val)
doc_2_val = np.array(doc_2_val)
top = doc_1_val.dot(doc_2_val)
bottom = np.linalg.norm(doc_1_val) * np.linalg.norm(doc_2_val)
cos1 = top/bottom
print(cos1)
print(doc_1_val.shape)
print(len(doc_1_val))


# In[11]:


# IDF
# Corpus size
Corpus = 2

# Document Frequency V1
doc = np.concatenate((doc_1_val, doc_2_val))
# print(doc)
# print(len(doc))
doc_2d = doc.reshape(2,int(len(doc)/2))
# print(doc_2d[0][4])
idf_1 = []
for i in range(len(doc_1_val)):
    df = 0
    for j in range(2):
#         print(doc_2d[i][j])
        df = df + doc_2d[j][i]
#         print(df)
        
    idf = np.log10(2/df)
    idf_1.append(idf)
print(idf_1)  

idf_2 = []
# Document Frequency V2
for i in zip(doc_1_val, doc_2_val):
    df = sum(i)
    idf = np.log10(2/df)
    idf_2.append(idf)
print(idf_2)  


# In[12]:


# tf_idf
# Document 1
tf_idf_1 = []
for i in range(len(idf_2)):
    tf_idf = doc_1_val[i] * idf_2[i]
    tf_idf_1.append(tf_idf)
print(tf_idf_1)

# Document 2
tf_idf_2 = []
for i in range(len(idf_2)):
    tf_idf = doc_2_val[i] * idf_2[i]
    tf_idf_2.append(tf_idf)
print(tf_idf_2)


# In[13]:


tf_idf_1 = np.array(tf_idf_1)
tf_idf_2 = np.array(tf_idf_2)
top = tf_idf_1.dot(tf_idf_2)
bottom = np.linalg.norm(tf_idf_1) * np.linalg.norm(tf_idf_2)
cos1 = top/(bottom + 1e-20)
print(cos1)


# In[14]:


Corpus = [Sentence_1,Sentence_2,Sentence_3,Sentence_4,Sentence_5]
print(Corpus)


# In[15]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()
print(tfidf_vectorizer)

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(Corpus)
print(tfidf_matrix)


# In[16]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_1 = Sent_1.copy()
print(Sent_1)

# for key,value in Sent_2.items():
#     akey = str(key)
# #     print(key)
#     if akey not in Sent_1.keys():
#         print(key)
#         doc_1[key] = 0
# #         S1 = Sent_1[str(key)] + Sent_2[str(key)]
# #         print(S1)
# #     print(key)
# #     print(value)

# print(doc_1)
print(sorted(doc_1.items()))

doc_1_val = []
for key,val in sorted(doc_1.items()):
    doc_1_val.append(val)
    
print(doc_1_val)


# In[17]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_2 = Sent_2.copy()
print(Sent_2)

# for key,value in Sent_2.items():
#     akey = str(key)
# #     print(key)
#     if akey not in Sent_1.keys():
#         print(key)
#         doc_1[key] = 0
# #         S1 = Sent_1[str(key)] + Sent_2[str(key)]
# #         print(S1)
# #     print(key)
# #     print(value)

# print(doc_1)
print(sorted(doc_2.items()))

doc_2_val = []
for key,val in sorted(doc_2.items()):
    doc_2_val.append(val)
    
print(doc_2_val)


# In[18]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_3 = Sent_3.copy()
print(Sent_3)

# for key,value in Sent_2.items():
#     akey = str(key)
# #     print(key)
#     if akey not in Sent_1.keys():
#         print(key)
#         doc_1[key] = 0
# #         S1 = Sent_1[str(key)] + Sent_2[str(key)]
# #         print(S1)
# #     print(key)
# #     print(value)

# print(doc_1)
print(sorted(doc_3.items()))

doc_3_val = []
for key,val in sorted(doc_3.items()):
    doc_3_val.append(val)
    
print(doc_3_val)


# In[19]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_4 = Sent_4.copy()
print(Sent_4)

# for key,value in Sent_2.items():
#     akey = str(key)
# #     print(key)
#     if akey not in Sent_1.keys():
#         print(key)
#         doc_1[key] = 0
# #         S1 = Sent_1[str(key)] + Sent_2[str(key)]
# #         print(S1)
# #     print(key)
# #     print(value)

# print(doc_1)
print(sorted(doc_4.items()))

doc_4_val = []
for key,val in sorted(doc_4.items()):
    doc_4_val.append(val)
    
print(doc_4_val)


# In[20]:


# doc_1 = Sent_1.keys()
# print(doc_1)

# for i in doc_1:
#     print(i)

doc_5 = Sent_5.copy()
print(Sent_5)

# for key,value in Sent_2.items():
#     akey = str(key)
# #     print(key)
#     if akey not in Sent_1.keys():
#         print(key)
#         doc_1[key] = 0
# #         S1 = Sent_1[str(key)] + Sent_2[str(key)]
# #         print(S1)
# #     print(key)
# #     print(value)

# print(doc_1)
print(sorted(doc_5.items()))

doc_5_val = []
for key,val in sorted(doc_5.items()):
    doc_5_val.append(val)
    
print(doc_5_val)


# In[21]:


# Document Frequency 
for i in zip(doc_1_val, doc_2_val):
    print(i)
    df = sum(i)
    idf = np.log10(2/df)
    idf_2.append(idf)
print(idf_2)  


# In[22]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()
print(tfidf_vectorizer)

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(Corpus)
print(tfidf_matrix)


# In[ ]:




