data_root = '/Users/lemon/downloads/project/'

import os
import time

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage.transform import resize
from skimage import io, data, color, exposure

def extract_hog_features(image_path):
    image = io.imread(image_path)
    image = color.rgb2gray(image)
    image_resized = resize(image, (256, 256))
    return hog(image_resized, orientations=8,
        pixels_per_cell=(16, 16), cells_per_block=(1, 1))

# extract image features and save it to .h5

# Initialize files
import h5py
f = h5py.File(data_root+'train_image_HOGfeatures.h5','w')
filenames = f.create_dataset('photo_id',(0,), maxshape=(None,),dtype='|S54')
feature = f.create_dataset('feature',(0,2048), maxshape = (None,2048))
f.close()

import pandas as pd
train_photos = pd.read_csv(data_root+'train_photo_to_biz_ids.csv')
train_folder = data_root+'train_photos/'
train_images = [os.path.join(train_folder, str(x)+'.jpg') for x in train_photos['photo_id']]  # get full filename

num_train = len(train_images)
print "Number of training images: ", num_train

tic = time.time()

# Training Images
for i in range(0, num_train):
    feature = extract_hog_features(train_images[i])
    num_done = i+1
    f= h5py.File(data_root+'train_image_HOGfeatures.h5','r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i] = train_images[i]
    f['feature'].resize((num_done,feature.shape[0]))
    f['feature'][i, :] = feature
    f.close()
    if num_done%10000==0 or num_done==num_train:
        print "Train images processed: ", num_done

toc = time.time()
print '\nFeatures extracted in %fs' % (toc - tic)