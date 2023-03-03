#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing
# 
# Chapter 20 of [Data Science from Scratch](http://shop.oreilly.com/product/0636920033400.do). Joel's code: [natural_language_processing.py](https://github.com/joelgrus/data-science-from-scratch/blob/master/code-python3/natural_language_processing.py)

# In[18]:


# python -m pip install beautifulsoup4 requests html5lib


# In[19]:


from bs4 import BeautifulSoup
from collections import defaultdict, Counter
import random
import requests
import re
import matplotlib.pyplot as plt


# In[20]:


# pip install "ipython-beautifulsoup[bs4]"
# pip install "ipython-beautifulsoup[bs4,notebook,qtconsole]"
# pip install bs4
# pip install "ipython-beautifulsoup[bs4]"


# ## n-gram models

# In[21]:


# url = 'http://radar.oreilly.com/2010/06/what-is-data-science.html'
# url = 'https://medium.com/@adriensieg/text-similarities-da019229c894'
url = 'https://www.oreilly.com/radar/what-is-data-science/'
# url = "https://www.oreilly.com/ideas/what-is-data-science"
html = requests.get(url).text


# In[22]:


soup = BeautifulSoup(html, 'html5lib')
content = soup.findAll('div', 'article-body')


# In[23]:


regex = r"[\w']+|[\.]"
document = []
for paragraph in soup.findAll("p"):
    words = re.findall(regex, paragraph.text)
    document.extend(words)


# In[24]:


bigrams = zip(document, document[1:])
transitions = defaultdict(list)
for prev, current in bigrams:
    transitions[prev].append(current)


# In[25]:


len(transitions)


# In[26]:


def generate_using_bigrams():
    # this means the next word will start a sentence
    current = "."
    result = []
    
    while True:
        next_word_candidates = transitions[current]
        current = random.choice(next_word_candidates)
        result.append(current)
        if current == ".": return " ".join(result)


# In[27]:


generate_using_bigrams()


# In[28]:


trigrams = zip(document, document[1:], document[2:])
trigram_transitions = defaultdict(list)
starts = []


# In[29]:


for a, b, c in trigrams:
    if a == ".":
        starts.append(b)
    trigram_transitions[(a, b)].append(c)


# In[30]:


def generate_using_trigrams():
    current = random.choice(starts)
    prev = "."
    result = [current]

    # choose a random starting word
    # and precede it with a '.'
    while True:
        next_word_candidates = trigram_transitions[(prev, current)]
        next_word = random.choice(next_word_candidates)
        prev, current = current, next_word
        result.append(current)
        
        if current == ".":
            return " ".join(result)


# In[31]:


generate_using_trigrams()


# ## Grammars

# In[32]:


grammar = {
    "_S"  : ["_NP _VP"],
    "_NP" : ["_N",
             "_A _NP _P _A _N"],
    "_VP" : ["_V",
             "_V _NP"],
    "_N"  : ["data science", "Python", "regression"],
    "_A"  : ["big", "linear", "logistic"],
    "_P"  : ["about", "near"],
    "_V"  : ["learns", "trains", "tests", "is"]
}


# In[33]:


def is_terminal(token):
    return token[0] != "_"


# In[34]:


def expand(grammar, tokens):
    for i, token in enumerate(tokens):

        # ignore terminals
        if is_terminal(token): continue

        # choose a replacement at random
        replacement = random.choice(grammar[token])

        if is_terminal(replacement):
            tokens[i] = replacement
        else:
            tokens = tokens[:i] + replacement.split() + tokens[(i+1):]
        return expand(grammar, tokens)

    # if we get here we had all terminals and are done
    return tokens


# In[35]:


def generate_sentence(grammar):
    return ' '.join(expand(grammar, ["_S"])) + '.'


# In[36]:


generate_sentence(grammar)


# ## Gibbs Sampling

# In[37]:


def roll_a_die():
    return random.choice([1,2,3,4,5,6])


# In[38]:


def direct_sample():
    d1 = roll_a_die()
    d2 = roll_a_die()
    return d1, d1 + d2


# In[39]:


td = [s for _, s in (direct_sample() for i in range(1000))]


# In[40]:


counter = Counter(td)
plt.bar(range(2,13), [counter[i] for i in range(2,13)])
plt.show()


# In[41]:


def random_y_given_x(x):
    """equally likely to be x + 1, x + 2, ... , x + 6"""
    return x + roll_a_die()


# In[42]:


def random_x_given_y(y):
    if y <= 7:
        # if the total is 7 or less, the first die is equally likely to be # 1, 2, ..., (total - 1)
        return random.randrange(1, y)
    else:
        # if the total is 7 or more, the first die is equally likely to be # (total - 6), (total - 5), ..., 6
        return random.randrange(y - 6, 7)


# In[43]:


def gibbs_sample(num_iters=100):
    x, y = 1, 2 # doesn't really matter
    for _ in range(num_iters):
        x = random_x_given_y(y)
        y = random_y_given_x(x)
    return x, y


# In[44]:


def compare_distributions(num_samples=10000):
    counts = defaultdict(lambda: [0, 0])
    for _ in range(num_samples):
        counts[gibbs_sample()][0] += 1
        counts[direct_sample()][1] += 1
    return counts


# In[45]:


counts = compare_distributions()


# In[46]:


for d1 in [1,2,3,4,5,6]:
    for d2 in [1,2,3,4,5,6]:
        print(d1,d1+d2,':',counts[(d1,d1+d2)])


# In[47]:


y = 7
p1 = plt.bar([x-0.2 for x in range(1,7)],
        [counts[(i,y)][0] for i in range(1,7)],
        color='#33669980',
        width=0.35)
p2 = plt.bar([x+0.2 for x in range(1,7)],
        [counts[(i,y)][1] for i in range(1,7)],
        color='#33993380',
        width=0.35)
plt.legend((p1[0], p2[0]), ('Gibbs Sampling', 'Direct Sampling'))
plt.show()


# ## Topic Modeling

# In[48]:


def sample_from(weights):
    """returns i with probability weights[i] / sum(weights)"""
    total = sum(weights)
    rnd = total * random.random()
    for i, w in enumerate(weights):
        rnd -= w
        if rnd <= 0: return i


# In[49]:


documents = [
        ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
        ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
        ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
        ["R", "Python", "statistics", "regression", "probability"],
        ["machine learning", "regression", "decision trees", "libsvm"],
        ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
        ["statistics", "probability", "mathematics", "theory"],
        ["machine learning", "scikit-learn", "Mahout", "neural networks"],
        ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
        ["Hadoop", "Java", "MapReduce", "Big Data"],
        ["statistics", "R", "statsmodels"],
        ["C++", "deep learning", "artificial intelligence", "probability"],
        ["pandas", "R", "Python"],
        ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
        ["libsvm", "regression", "support vector machines"]
]


# In[50]:


document_topic_counts = [Counter() for _ in documents]


# In[51]:


K = 4
topic_word_counts = [Counter() for _ in range(K)]


# In[52]:


topic_counts = [0 for _ in range(K)]


# In[53]:


document_lengths = list(map(len, documents))


# In[54]:


distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)


# In[55]:


D = len(documents)


# In[56]:


document_topic_counts[3][1]


# In[57]:


topic_word_counts[2]["nlp"]


# In[58]:


def p_topic_given_document(topic, d, alpha=0.1):
    """the fraction of words in document _d_
    that are assigned to _topic_ (plus some smoothing)"""
    return ((document_topic_counts[d][topic] + alpha) /
            (document_lengths[d] + K * alpha))


# In[59]:


def p_word_given_topic(word, topic, beta=0.1):
    """the fraction of words assigned to _topic_
    that equal _word_ (plus some smoothing)"""
    return ((topic_word_counts[topic][word] + beta) /
            (topic_counts[topic] + W * beta))


# In[60]:


def topic_weight(d, word, k):
    """given a document and a word in that document,
    return the weight for the kth topic"""
    return p_word_given_topic(word, k) * p_topic_given_document(k, d)


# In[61]:


def choose_new_topic(d, word):
    return sample_from([topic_weight(d, word, k)
        for k in range(K)])


# In[62]:


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
    for document in documents]


# In[63]:


for d in range(D):
    for word, topic in zip(documents[d], document_topics[d]):
        document_topic_counts[d][topic] += 1
        topic_word_counts[topic][word] += 1
        topic_counts[topic] += 1


# In[64]:


for iter in range(1000):
    for d in range(D):
        for i, (word, topic) in enumerate(zip(documents[d],
                                              document_topics[d])):

            # remove this word / topic from the counts
            # so that it doesn't influence the weights
            document_topic_counts[d][topic] -= 1
            topic_word_counts[topic][word] -= 1
            topic_counts[topic] -= 1
            document_lengths[d] -= 1
            
            # choose a new topic based on the weights
            new_topic = choose_new_topic(d, word)
            document_topics[d][i] = new_topic
            
            # and now add it back to the counts
            document_topic_counts[d][new_topic] += 1
            topic_word_counts[new_topic][word] += 1
            topic_counts[new_topic] += 1
            document_lengths[d] += 1


# In[65]:


for k, word_counts in enumerate(topic_word_counts):
    for word, count in word_counts.most_common():
        if count > 0:
            print(k, word, count)


# In[66]:


for document, topic_counts in zip(documents, document_topic_counts):
    print(document)
    for topic, count in topic_counts.most_common():
        if count > 0:
            print(topic, ':', count, '\n')


# In[ ]:




