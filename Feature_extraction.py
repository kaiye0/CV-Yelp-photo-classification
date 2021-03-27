caffe_root = '/Users/lemon/caffe/'
data_root = '/Users/lemon/downloads/project/'

import numpy as np
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe
import os

if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print("Downloading pre-trained CaffeNet model...")
    # caffe_root/scripts/download_model_binary.py ../models/bvlc_reference_caffenet


def extract_features(images, layer='fc7'):
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
        1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB]]

    num_images = len(images)
    net.blobs['data'].reshape(num_images, 3, 227, 227)
    net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', caffe.io.load_image(x)), images)
    out = net.forward()

    return net.blobs[layer].data


# extract image features and save it to .h5

# Initialize files
import h5py

# f.close()
f = h5py.File(data_root+'train_image_fc7features.h5','w')
filenames = f.create_dataset('photo_id', (0,), maxshape=(None,), dtype='|S54')
feature = f.create_dataset('feature', (0, 4096), maxshape=(None, 4096))
f.close()

import pandas as pd

train_photos = pd.read_csv(data_root + 'train_photo_to_biz_ids.csv')
train_folder = data_root + 'train_photos/'
train_images = [os.path.join(train_folder, str(x) + '.jpg') for x in train_photos['photo_id']]  # get full filename

num_train = len(train_images)
print "Number of training images: ", num_train
batch_size = 500

# Training Images
for i in range(0, num_train, batch_size):
    images = train_images[i: min(i + batch_size, num_train)]
    features = extract_features(images, layer='fc7')
    num_done = i + features.shape[0]
    f = h5py.File(data_root + 'train_image_fc7features.h5', 'r+')
    f['photo_id'].resize((num_done,))
    f['photo_id'][i: num_done] = np.array(images)
    f['feature'].resize((num_done, features.shape[1]))
    f['feature'][i: num_done, :] = features
    f.close()
    if num_done % 20000 == 0 or num_done == num_train:
        print "Train images processed: ", num_done
