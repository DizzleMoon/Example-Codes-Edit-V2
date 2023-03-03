#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile
import shutil
import os


# In[3]:


def copy_file(directory, *filenames):
    
    for filename in filenames:
        try:
            isFile = os.path.isfile(filename)
            print(isFile)
            shutil.copy(filename, directory)
        except:
            pass
    return


# In[ ]:


def zip_py(directory, zip_directory, mode, *filenames):
    
    filed = copy_file("insert directory"," insert file")_dox
    
    ## Write
    if mode == 'w':
        zip_doc = zipfile.ZipFile(zip_directory, mode = 'w')
        for filename in filenames:
            zip_doc.write(filename)
            
    ## Read
    elif mode == 'r':
        zip_doc = zipfile.ZipFile(zip_directory, mode = 'r')
        for filename in filenames:
            print("\n", zip_doc.read(filename))
            
    ## Extract one at a time
    elif mode == 'extract':
        if mode == 'extract':
            zip_doc = zipfile.ZipFile(zip_directory, mode = 'r')
        for filename in filenames:
            print("\n", zip_doc.extract(filename))
            
    ## Extract all
    elif mode == 'extract all':
        zip_doc = zipfile.ZipFile(zip_directory, mode = 'r')
        zip_doc.extractall()
        
    zip_doc.close()
    
    return
 

