import pickle
import numpy as np
from scipy import ndimage
from PIL import Image


augmented_file = "d:/archi/ernesto/cursos/self-driving car/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train_aug.p"

with open(augmented_file, mode='rb') as f:
    train = pickle.load(f)

x, y = train['features'], train['labels']
freqs = {key: 0 for key in y}


for index in range(len(y)):
    freqs[y[index]] += 1

total = 0
for key, value in freqs.items():
    print("%d: %d" % (key, value))
    total += value

print("Total: ", total)

