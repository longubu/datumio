"""
Test datagen generators

Concusion: 
--------
    - Do dtf.build_augmentation_transform() then dtf.transform_image(img, tf=tf)
    - Do [], and append to list when doing batch. np.zeros(shape) 
        is same speed, but requires more knowledge of batch sizes
    - Data loading in batch + batch transformations is same speed as 
        loading individual data and tf on individual images
        
"""
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt
import time

import load_cifar_10
import datumio.transforms as dtf

X, y = load_cifar_10.load_cifar10_data()
augmentation_params = dict(
        rotation = 15,
        zoom = (1.5, 1.5), # zoom (x, y) = (col, row)
        shear = 7,
        translation = (5, -5),
        flip_lr = True,
        )


t = time.time()
input_shape = X.shape[1:3]
tf = dtf.build_augmentation_transform(input_shape, **augmentation_params)
images = np.zeros(X.shape)
for it, img in enumerate(X):
    images[it] = dtf.transform_image(img, tf=tf)
et = time.time()
print '... Took %s to finish for loop transform using pre-built tf & transform'%(et - t)

t = time.time()
input_shape = X.shape[1:3]
tf = dtf.build_augmentation_transform(input_shape, **augmentation_params)
images = np.zeros(X.shape)
for it, img in enumerate(X):
    images[it] = dtf.fast_warp(img, tf)
et = time.time()
print '... Took %s to finish for loop transform using pre-built tf & fastwarp'%(et - t)

t = time.time()
images = []
for img in X:
    images.append(dtf.transform_image(img, **augmentation_params))
images = np.array(images)
et = time.time()
print '... Took %s to finish for loop transform function'%(et - t)

t = time.time()
images = np.zeros(X.shape)
for it, img in enumerate(X):
    images[it] = dtf.transform_image(img, **augmentation_params)
et = time.time()
print '... Took %s to finish for loop transform using np array'%(et - t)

# fastest #1
t = time.time()     
X_new = dtf.transform_images(X, tf_image_kwargs=augmentation_params)
et = time.time()
print '... Took %s to finish batch transform function using **augmentaiton'%(et - t)

t = time.time()
input_shape = X.shape[1:3]
tf = dtf.build_augmentation_transform(input_shape, **augmentation_params)  
X_new = dtf.transform_images(X, tf_image_kwargs=dict(tf=tf))
et = time.time()
print '... Took %s to finish batch transform function using tf=tf'%(et - t)

# test speed with dataloading
dataPath = 'test_data/cat.jpg'
nTimes = 100

t = time.time()
t_imgs = []
for x in xrange(nTimes):
    img = np.array(Image.open(dataPath))
    t_imgs += [dtf.transform_image(img, **augmentation_params)]
et = time.time()
print '... Took %s to finish loading & tf on individual images'%(et - t)

t = time.time()
imgs = []
for x in xrange(nTimes):
    imgs += [np.array(Image.open(dataPath))]
imgs = np.array(imgs)
t_imgs = dtf.transform_images(imgs, tf_image_kwargs=augmentation_params)
et = time.time()
print '... Took %s to finish loading & tf on batch images'%(et - t)