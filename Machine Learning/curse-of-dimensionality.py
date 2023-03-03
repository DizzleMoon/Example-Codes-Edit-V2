#!/usr/bin/env python
# coding: utf-8

# In[8]:


import math
import random


# In[24]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


def distance(a, b):
    return math.sqrt(sum((ai-bi)**2 for ai,bi in zip(a,b)))


# In[19]:


def mean(xs):
    return sum(xs)/len(xs)


# In[20]:


def random_point(d):
    return tuple(random.random() for _ in range(d))


# In[21]:


def random_distances(d, num_pairs):
    return [distance(random_point(d), random_point(d))
            for _ in range(num_pairs)]


# In[28]:


avg_distances = []
min_distances = []

dimensions = list(range(1,101))

for dim in dimensions:
    ds = random_distances(dim, 1000)
    avg_distances.append(mean(ds))
    min_distances.append(min(ds))


# In[30]:


plt.plot(dimensions, avg_distances)
plt.plot(dimensions, min_distances)
plt.show()


# “As the number of dimensions increases, the average distance between points increa‐ ses. But what’s more problematic is the ratio between the closest distance and the average distance.”

# In[31]:


min_avg_ratio = [min_dist / avg_dist 
                 for min_dist, avg_dist in zip(min_distances, avg_distances)]


# In[32]:


plt.plot(dimensions, min_avg_ratio)
plt.show()


# “In low-dimensional data sets, the closest points tend to be much closer than average. But two points are close only if they’re close in every dimension, and every extra dimension—even if just noise—is another opportunity for each point to be further away from every other point. When you have a lot of dimensions, it’s likely that the closest points aren’t much closer than average, which means that two points being close doesn’t mean very much (unless there’s a lot of structure in your data that makes it behave as if it were much lower-dimensional).”

# In[ ]:




