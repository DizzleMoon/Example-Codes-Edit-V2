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
 
# print(Sent_1['pizza'])


# In[4]:


wordset = set(Sen_1 ).union(set(Sen_2)).union(set(Sen_3)).union(set(Sen_4)).union(set(Sen_5))
wordset


# In[5]:


# Dictinonary for each document
doc_1  = dict.fromkeys(wordset,0)
doc_2  = dict.fromkeys(wordset,0)
doc_3  = dict.fromkeys(wordset,0)
doc_4  = dict.fromkeys(wordset,0)
doc_5  = dict.fromkeys(wordset,0)


# In[6]:


# Count the words
for word in Sen_1:
    doc_1[word] += 1
    
for word in Sen_2:
    doc_2[word] += 1
    
for word in Sen_3:
    doc_3[word] += 1
    
for word in Sen_4:
    doc_4[word] += 1
    
for word in Sen_5:
    doc_5[word] += 1


# In[7]:


# Create DataFrame
docs = pd.DataFrame([doc_1,doc_2,doc_3,doc_4,doc_5])
print(docs)

print('\n')

# # Column names
# for col in docs.columns:
#     print(col)
    
# print('\n')


# # Row names
# for row in range(len(docs)):
#     print(row)
#     print(docs.iloc[row])
    
# print('\n')

# # Num of occurences for column title 'the'
# print(docs['the'])


# In[8]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

Corpus = [Sentence_1,Sentence_2,Sentence_3,Sentence_4,Sentence_5]
# print(Corpus)

# Initialize an instance of tf-idf Vectorizer
tfidf_vectorizer = TfidfVectorizer()
print(tfidf_vectorizer)

# Generate the tf-idf vectors for the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(Corpus)
print(tfidf_matrix)
print(tfidf_matrix.shape)

# compute and print the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
print(cosine_sim)


# In[ ]:




