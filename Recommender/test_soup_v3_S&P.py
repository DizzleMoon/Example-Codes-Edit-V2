#!/usr/bin/env python
# coding: utf-8

# In[11]:


#Python program to scrape website 
#and save quotes from website
import requests as r
from bs4 import BeautifulSoup
import csv
import bs4 as bs
import urllib.request
import re
import time


# In[12]:


# headers = requests.utils.default_headers()
# # headers: {
# #     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
# #     'accept-encoding': 'gzip, deflate, br',
# #     'accept-language': 'en-US,en;q=0.9,fr;q=0.8,ro;q=0.7,ru;q=0.6,la;q=0.5,pt;q=0.4,de;q=0.3',
# #     'cache-control': 'max-age=0',
# #     'upgrade-insecure-requests': '1',
# #     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# # }

# headers = {
#     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36',
#     'referrer': 'https://google.com',
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#     'Accept-Encoding': 'gzip, deflate, br',
#     'Accept-Language': 'en-US,en;q=0.9',
#     'Pragma': 'no-cache'}


# In[13]:


# source = urllib.request.urlopen('https://www.bloomberg.com/markets/stocks')
# source = urllib.request.urlopen('https://www.skysports.com/football')\
url = 'https://www.bloomberg.com/quote/SPX:IND'
# url = 'https://www.bloomberg.com/markets/stocks'
# url = 'https://www.bloomberg.com/energy'
f = urllib.request.Request(url, headers={"user-agent": "Mozilla/80.0"})
source = urllib.request.urlopen(f)
# quote_page = 'https://www.bloomberg.com/markets/stocks'
# source = r.get(quote_page, headers={"user-agent": "Mozilla/80.0"})


# In[14]:


# page_data = soup(data.text, 'html5lib')


# In[15]:


# headers = requests.utils.default_headers()
# headers: {
#     'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#     'accept-encoding': 'gzip, deflate, br',
#     'accept-language': 'en-US,en;q=0.9,fr;q=0.8,ro;q=0.7,ru;q=0.6,la;q=0.5,pt;q=0.4,de;q=0.3',
#     'cache-control': 'max-age=0',
#     'upgrade-insecure-requests': '1',
#     'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'
# }


# In[16]:


soup = bs.BeautifulSoup(source,'html.parser')


# In[17]:


soup.find('div', class_='price-content')
# soup.prettify


# In[18]:


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


# In[19]:


# # print(soup.findAll('tr'))[14]
# brent = soup.findAll('tr')
# BloomPrice = float(re.search(re.compile (r'\d+\.\d*'),str(soup.findAll('tr').contents)).group())


# In[20]:


prices = soup.find_all(class_='priceText__1853e8a5')
# prices[0].text


# In[21]:


for paragraph in soup.findAll('p'):
    print(paragraph.string)
    print(str(paragraph.text))


# In[22]:


# bloomberg = soup.get_text()
# print(bloomberg)
# brent = soup.find('tr')
# str(brent.text)


# In[23]:


# import requests
# from bs4 import BeautifulSoup

# response = requests.get('https://www.bloomberg.com/quote/SPX:IND')
# soup = BeautifulSoup(response.text, 'lxml')
# price = soup.select_one('.price')
# print(soup.select_one('.price'))


# In[24]:


# from bs4 import BeautifulSoup
# from requests import Session

# session = Session()
# session.headers['user-agent'] = (
#     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
#     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'
#     '66.0.3359.181 Safari/537.36'
# )

# quote_page = 'https://www.bloomberg.com/quote/SPX:IND'

# page= session.get(quote_page)

# soup = BeautifulSoup(page.text, 'html.parser')

# price_box = soup.find('meta', itemprop="price")

# price = float(price_box)

# print(price)


# In[25]:


# import csv
# from bs4 import BeautifulSoup
# import requests

# url = 'https://www.bloomberg.com/markets/stocks'
# headers = ('Name', 'Value', 'Net Change', '% Change' '1 Month', '1 Year', 'Time (EDT)')

# r = requests.get(url)
# soup = BeautifulSoup(r.text, 'lxml')

# trs = soup.select('.data-table-body > tr')

# print(trs)


# # with open('data.csv', 'w') as outcsv:
# #     writer = csv.writer(outcsv)
# #     writer.writerow(headers)

# #     for tr in trs:
# #         tds = tr.find_all('td')[:7]
# #         tds[0].select_one('[data-type="abbreviation"]').decompose()  # optional

# #         content = [td.text.strip() for td in tds]
# #         writer.writerow(content)


# In[26]:


# import requests
# from bs4 import BeautifulSoup as bs

# quote_page ='https://www.bloomberg.com/quote/SPX:IND'
# page = requests.get(quote_page, headers = {'User-Agent':'Mozilla/5.0', 'accept-language':'en-US,en;q=0.9'})
# soup = bs(page.content,'lxml')
# name_box = soup.select_one('[class^=companyName]')
# name = name_box
# print(soup.select_one('[class^=companyName]'))


# In[27]:


# # get the index price
# price_box = soup.find('div', attrs={'class':'price'})
# price = price_box.text
# print(price)

