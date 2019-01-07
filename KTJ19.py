
# coding: utf-8

# In[14]:


fileInput =  "dataset.json"
fileOutput = "output.csv"


# In[20]:


inputFile = open(fileInput, 'r', encoding="utf8") #open json file
outputFile = open(fileOutput, 'w', encoding='utf8') #load csv file


# In[21]:


data = json.load(inputFile) #load json content
inputFile.close() #close the input file


# In[22]:


output = csv.writer(outputFile) #create a csv.write


# In[23]:


output.writerow(data[0].keys())  # header row


# In[24]:


for row in data:
        output.writerow(row.values()) #values row


# In[25]:


import pandas as pd
data=pd.read_csv('output.csv')


# In[28]:


cat=data['category']


# In[29]:


len(set(cat))


# In[30]:


# There are 41 unique output categories in the given data set.

