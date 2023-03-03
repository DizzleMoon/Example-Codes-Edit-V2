#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Python program to scrape website 
#and save quotes from website
import requests 
from bs4 import BeautifulSoup
import csv
import bs4 as bs
import urllib.request
import re
import time


# In[2]:


def fix_unicode(text: str) -> str:
    return text.replace(u"\u2019", "'")


# In[3]:


# url = 'http://radar.oreilly.com/2010/06/what-is-data-science.html'
# url = 'https://medium.com/@adriensieg/text-similarities-da019229c894'
# url = 'https://www.oreilly.com/radar/what-is-data-science/'
# url = "https://www.oreilly.com/ideas/what-is-data-science"
url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"
# html = r.get(url).text
f = urllib.request.Request(url, headers={"user-agent": "Mozilla/80.0"})
source = urllib.request.urlopen(f)


# In[4]:


soup = bs.BeautifulSoup(source,'html.parser')


# In[5]:


soup.find('div', class_='article-body')
soup.prettify
# content = soup.find_all(class_='article-body')
# content


# In[6]:


# title of the page
print(soup.title)

# get attributes:
print(soup.title.name)

# get values:
print(soup.title.string)

# beginning navigation:
print(soup.title.parent.name)

# getting specific values:
# print(soup.p)

# print(soup.headers)

# print(soup.content)

# print(soup.find('p',id="bDescTeaser"))


# In[7]:


for paragraph in soup.findAll('p'):
    print(paragraph.string)
    print(str(paragraph.text))


# In[8]:


first_paragraph = soup.find('p')


# In[9]:


first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()


# In[10]:


first_paragraph_id = soup.p['id']       # raises KeyError if no 'id'
first_paragraph_id2 = soup.p.get('id')  # returns None if no 'id'


# In[11]:


all_paragraphs = soup.find_all('p')  # or just soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]


# In[12]:


important_paragraphs = soup('p', {'class' : 'important'})
important_paragraphs2 = soup('p', 'important')
important_paragraphs3 = [p for p in soup('p')
                         if 'important' in p.get('class', [])]


# In[13]:


# Warning: will return the same <span> multiple times
# if it sits inside multiple <div>s.
# Be more clever if that's the case.
spans_inside_divs = [span for div in soup('div')     # for each <div> on the page
                     for span in div('span')]   # find each <span> inside it


# In[14]:


url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href']for a in soup('a') if a.has_attr('href')]

print(len(all_urls))  # 965 for me, way too many


# In[15]:


# Must start with http:// or https://
# Must end with .house.gov or .house.gov/
regex = r"^https?://.*\.house\.gov/?$"


# In[16]:


# And now apply
good_urls = [url for url in all_urls if re.match(regex, url)]

print(len(good_urls))  # still 862 for me


# In[17]:


good_urls = list(set(good_urls))

print(len(good_urls))  # only 431 for me


# In[18]:


# url = 'http://radar.oreilly.com/2010/06/what-is-data-science.html'
# url = 'https://medium.com/@adriensieg/text-similarities-da019229c894'
# url = 'https://www.oreilly.com/radar/what-is-data-science/'
url = "https://www.oreilly.com/ideas/what-is-data-science"
# url = "https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html"
# html = r.get(url).text
f = urllib.request.Request(url, headers={"user-agent": "Mozilla/80.0"})
# f = urllib.request.Request(url)
source = urllib.request.urlopen(f)


# In[19]:


soup = bs.BeautifulSoup(source,'html.parser')


# In[20]:


soup.find('div', class_='article-body')
soup.prettify
# content = soup.find_all(class_='article-body')
# content


# In[21]:


# title of the page
print(soup.title)

# get attributes:
print(soup.title.name)

# get values:
print(soup.title.string)

# beginning navigation:
print(soup.title.parent.name)

# getting specific values:
print(soup.p)

print(soup.headers)

print(soup.content)

print(soup.find('p',id="bDescTeaser"))


# In[22]:


for paragraph in soup.findAll('p'):
#     print(paragraph.string)
    print(str(paragraph.text))


# In[23]:


soup.find('meta', {'content': '"What is data science?"'})
# soup.prettify


# In[24]:


soup.prettify


# In[25]:


# para = soup.find_all('p')
# para[1]
# para[1].text.find('according')


# In[26]:


content = soup.findAll('div','article')
regex = re.compile(r"[\w|']+|[\.?,s\']")
regex1 = re.compile(r"[\w|']+|[\.?,\S\']")


# In[27]:


document = []
 
for paragraph in soup.findAll('p'):
    print('para:',paragraph)
    words = re.findall(regex1, paragraph.text)
#     para = fix_unicode(paragraph.text)
#     words = re.findall(regex1, fix_unicode(para))
    document.extend(words)


# In[28]:


document


# In[29]:


document.index('according')


# In[30]:


re.split("[']", "pete - he's a boy")


# In[31]:


target_string = "pete - he's a boy"
# re.compile(r"[\w|']+|[\.?,s\']")
result = re.split(r"[\W+\s]", target_string)
print(result)


# In[ ]:





# In[ ]:




