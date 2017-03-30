
# coding: utf-8

# ## Display a sample image from training set along with its grayscale and rotated variations

# In[1]:

import pickle
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import misc
from sklearn.utils import shuffle


# In[2]:

# general-purpose functions
get_ipython().magic('matplotlib inline')

def display_image(img, label):
    # visualize an image and its label
    if len(img.shape) > 2:
        img = img.squeeze()
    plt.figure(figsize=(1, 1))
    plt.imshow(img, cmap='gray')
    #plt.show()
    print(label)

def display_image_vector(v, label):
    rows = 1
    cols = len(v)
    fig = plt.figure(figsize=(cols, rows))
    print(label)
    for l in range(rows):
        for i in range(1, cols + 1):
            fig.add_subplot(rows, cols, i + l * cols)
            plt.axis('off')
            plt.imshow(v[i-1].squeeze())
    #plt.show()


# In[3]:

train_file = "./traffic-signs-data/train.p"
train_file_preproc = "./traffic-signs-data/train_preproc.p"
with open(train_file, mode='rb') as f:
    train = pickle.load(f)
with open(train_file_preproc, mode='rb') as f:
    train_preproc = pickle.load(f)


# In[21]:

index = random.randint(0, len(train['features']))
v = [train['features'][index], 
     train_preproc['features'][index], 
     train_preproc['features'][index+len(train)],
     train_preproc['features'][index+(2*len(train))]]
display_image_vector(v,"Train image #%d" %index)


# In[ ]:



