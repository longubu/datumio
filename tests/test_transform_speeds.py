"""
Test datagen generators
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


#fastest #2
t = time.time()
input_shape = X.shape[1:3]
tf = dtf.build_augmentation_transform(input_shape, **augmentation_params)
images = np.zeros(X.shape)
for it, img in enumerate(X):
    images[it] = dtf.transform_image(img, tf=tf)
et = time.time()
print '... Took %s to finish for loop transform using pre-built tf & transform'%(et - t)

#fastest #2
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


