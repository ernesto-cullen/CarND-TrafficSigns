
# coding: utf-8

# ## Visualize the training set

# In[17]:

import pickle
import numpy as np
from scipy import ndimage
from PIL import Image

# load images and labels
train_file = "./traffic-signs-data/train.p"

with open(train_file, mode='rb') as f:
    train = pickle.load(f)

x, y = train['features'], train['labels']
freqs = {key: [0,[]] for key in y}


for index in range(len(y)):
    value = y[index]
    freqs[value][0] += 1
    freqs[value][1].append(index)


# In[18]:

#load codification for labels
import csv
signnames_file = "./signnames.csv"
labels = {}
with open(signnames_file, mode='r') as f:
    rows = csv.reader(f, delimiter=',')
    for code,desc in rows:
        if code == 'ClassId':
            continue
        labels[code] = desc


# In[19]:

# distribution of classes
# output formatted for a markup table
total = 0
print("|code|description|# of images|")
print("|---|---|---:|")
for key, value in freqs.items():
    print("| %d | %s | %d |" % (key, labels[str(key)], value[0]))
    total += value[0]

print("Total: ", total)


# In[20]:

## display 10 examples for each class
import matplotlib.pyplot as plt
import random
from scipy import misc

# Visualizations will be shown in the notebook.
get_ipython().magic('matplotlib inline')

rows = 43
cols = 10
fig = plt.figure(figsize=(cols, rows))
for l in range(rows):
    #plt.xlabel("%d: %s" % (l, labels[l]))
    for i in range(1, cols+1):
        rand = random.randint(0,100)
        #print("l: %d, i: %d, rand: %d" % (l, i, rand))
        fig.add_subplot(rows, cols, i+l*cols)
        plt.axis('off')
        plt.imshow(x[freqs[l][1][rand]].squeeze())


# In[ ]:



