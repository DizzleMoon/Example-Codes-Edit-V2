#!/usr/bin/env python
# coding: utf-8

# # Decision Trees
# 
# Chapter 17 of _Data Science from Scratch_.

# In[1]:


import math
import random
from collections import Counter, defaultdict
from functools import partial


# ## Entropy

# In[2]:


def entropy(class_probabilities):
    """given a list of class probabilities, compute the entropy"""
    return sum(-p * math.log(p, 2)
               for p in class_probabilities
               if p)


# In[3]:


def class_probabilities(labels):
    total_count = len(labels)
    return [count / total_count for count in Counter(labels).values()]


# In[4]:


def data_entropy(labeled_data):
    labels = [label for _, label in labeled_data]
    probabilities = class_probabilities(labels)
    return entropy(probabilities)


# ## Entropy of some test strings
# 
# Just for kicks, let's check on the entropy of some english, plus some HTML.

# In[5]:


entropy(class_probabilities('This is just a regular english sentence.'))


# In[6]:


a = [random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz</>.!?,";') for i in range(174378)]


# In[7]:


entropy(class_probabilities(a))


# In[8]:


import requests
f = requests.get('http://shakespeare.mit.edu/midsummer/full.html')
msnd = f.text
len(msnd)


# In[9]:


entropy(class_probabilities(msnd))


# ## Building a decision tree

# In[10]:


inputs = [
    ({'level':'Senior','lang':'Java','tweets':'no','phd':'no'},   False),
    ({'level':'Senior','lang':'Java','tweets':'no','phd':'yes'},  False),
    ({'level':'Mid','lang':'Python','tweets':'no','phd':'no'},     True),
    ({'level':'Junior','lang':'Python','tweets':'no','phd':'no'},  True),
    ({'level':'Junior','lang':'R','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'R','tweets':'yes','phd':'yes'},    False),
    ({'level':'Mid','lang':'R','tweets':'yes','phd':'yes'},        True),
    ({'level':'Senior','lang':'Python','tweets':'no','phd':'no'}, False),
    ({'level':'Senior','lang':'R','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'yes','phd':'no'}, True),
    ({'level':'Senior','lang':'Python','tweets':'yes','phd':'yes'},True),
    ({'level':'Mid','lang':'Python','tweets':'no','phd':'yes'},    True),
    ({'level':'Mid','lang':'Java','tweets':'yes','phd':'no'},      True),
    ({'level':'Junior','lang':'Python','tweets':'no','phd':'yes'},False)
]

inputs


# In[11]:


def partition_entropy(subsets):
    """
    find the entropy from this partition of data into subsets
    
    subsets is a list of lists of labeled data
    """
    total_count = sum(len(subset) for subset in subsets)
    return sum( data_entropy(subset) * len(subset) / total_count for subset in subsets )


# In[12]:


def partition_by(inputs, attribute):
    """returns a dict of inputs partitioned by the attribute
    each input is a pair (attribute_dict, label)"""
    return group_by(inputs, lambda x: x[0][attribute])


# In[13]:


def partition_entropy_by(inputs, attribute):
    """computes the entropy corresponding to the given partition"""
    partitions = partition_by(inputs, attribute)
    return partition_entropy(partitions.values())


# In[14]:


def group_by(items, key_fn):
    """returns a defaultdict(list), where each input item
    is in the list whose key is key_fn(item)"""
    groups = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)
    return groups


# In[15]:


for key in ['level','lang','tweets','phd']:
    print(key, partition_entropy_by(inputs, key))


# In[16]:


gb_level = partition_by(inputs, 'level')
for key, items in gb_level.items():
    print('{:12} {} {}'.format(key, sum(1 for item in items if item[1]), sum(1 for item in items if not item[1])))


# In[17]:


for key in ['lang','tweets','phd']:
    print(key, partition_entropy_by(gb_level['Senior'], key))


# In[18]:


gb_level = partition_by(inputs, 'lang')
for key, items in gb_level.items():
    print('{:12} {} {}'.format(key, sum(1 for item in items if item[1]), sum(1 for item in items if not item[1])))


# In[19]:


def classify(tree, input):
    """classify the input using the given decision tree"""

    # if this is a leaf node, return its value
    if tree in [True, False]:
        return tree

    # otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values of are subtrees to consider next
    attribute, subtree_dict = tree

    subtree_key = input.get(attribute)  # None if input is missing attribute

    if subtree_key not in subtree_dict: # if no subtree for key,
        subtree_key = None              # we'll use the None subtree

    subtree = subtree_dict[subtree_key] # choose the appropriate subtree
    return classify(subtree, input)     # and use it to classify the input


# In[20]:


def build_tree_id3(inputs, split_candidates=None):

    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    # count Trues and Falses in the inputs
    num_trues = len([label for item, label in inputs if label])
    num_falses = len(inputs) - num_trues
    
    if num_trues == 0:
        return False
    
    if num_falses == 0:
        return True
    
    if not split_candidates:
        return num_trues >= num_falses

    # otherwise, split on the best attribute
    best_attribute = min(split_candidates,
        key=partial(partition_entropy_by, inputs))
    
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    
    # recursively build the subtrees
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.items() }
    
    subtrees[None] = num_trues > num_falses # default case
    
    return (best_attribute, subtrees)


# In[21]:


tree = build_tree_id3(inputs)


# In[22]:


classify(tree, {'level':'Junior','lang':'Python','tweets':'yes','phd':'yes'})


# In[23]:


tree


# In[24]:


classify(tree, { "level" : "Junior",
                 "lang" : "Java",
                 "tweets" : "yes",
                 "phd" : "no"} )


# In[25]:


classify(tree, { "level" : "Junior",
                 "lang" : "Java",
                 "tweets" : "yes",
                 "phd" : "yes"} )


# In[26]:


classify(tree, { "level" : "Intern" } )


# In[27]:


classify(tree, { "level" : "Senior" } )


# ## Random Forest

# In[28]:


def forest_classify(trees, input):
    votes = [classify(tree, input) for tree in trees]
    vote_counts = Counter(votes)
    print(votes)
    return vote_counts.most_common(1)[0][0]


# In[29]:


def build_tree_id3(inputs, split_candidates=None, num_split_candidates=None):

    # if this is our first pass,
    # all keys of the first input are split candidates
    if split_candidates is None:
        split_candidates = inputs[0][0].keys()

    
    # count Trues and Falses in the inputs
    num_trues = len([label for item, label in inputs if label])
    num_falses = len(inputs) - num_trues
    
    if num_trues == 0:
        return False
    
    if num_falses == 0:
        return True
    
    if not split_candidates:
        return num_trues >= num_falses
    
    # if there's already few enough split candidates, look at all of them
    if num_split_candidates is None or len(split_candidates) <= num_split_candidates:
        sampled_split_candidates = split_candidates    
    # otherwise pick a random sample
    else:
        sampled_split_candidates = random.sample(split_candidates, num_split_candidates)

    # otherwise, split on the best attribute
    best_attribute = min(sampled_split_candidates,
        key=partial(partition_entropy_by, inputs))
    
    partitions = partition_by(inputs, best_attribute)
    new_candidates = [a for a in split_candidates if a != best_attribute]
    
    # recursively build the subtrees
    subtrees = { attribute : build_tree_id3(subset, new_candidates)
                 for attribute, subset in partitions.items() }
    
    subtrees[None] = num_trues > num_falses # default case
    
    return (best_attribute, subtrees)


# In[30]:


def build_forest(inputs, n=3):
    return [build_tree_id3(inputs, num_split_candidates=3) for i in range(n)]


# In[31]:


trees = build_forest(inputs)
type(trees)


# In[32]:


forest_classify(trees, { "level" : "Senior",
                 "lang" : "Python",
                 "tweets" : "no",
                 "phd" : "no"})


# In[33]:


trees


# In[ ]:




