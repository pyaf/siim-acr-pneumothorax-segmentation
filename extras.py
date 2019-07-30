import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from image_utils import *


train_df = pd.read_csv('data/train32.csv')
fnames = train_df['id_code'].values
#size = 300
root = 'data/train_images/bgcc300/'
images = []
for fname in tqdm(fnames):
    path = os.path.join(root, fname + ".npy")
    #image = load_ben_color(path, size=size, crop=True)
    image = np.load(path)
    images.append(image)

np.save('data/all_train32_bgcc300.npy', images)

'''
train_df = pd.read_csv('data/test.csv')
fnames = train_df['id_code'].values
size = 300
root = 'data/test_images'
images = []
for fname in tqdm(fnames):
    path = os.path.join(root, fname + ".png")
    image = load_ben_color(path, size=size, crop=True)
    images.append(image)

np.save('data/all_test_bgcc300.npy', images)

'''
