"""
Test datagen generators
"""
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

import datumio.datagen as dtd

# load data & labels
dataDir = 'test_data/cifar-10/'
labelPath = 'test_data/cifar-10/labels.csv'
labelDF = pd.read_csv(labelPath)
uid_labels = labelDF.values
input_shape = (32, 32)
X = []
for uid, label in uid_labels:
    img = Image.open(os.path.join(dataDir, uid))
    X += [np.array(img)]
X = np.array(X, dtype = np.uint8)
y = np.array(uid_labels[:, 1], dtype = int)

# set up batch generator
batch_size = 32
batchgen = dtd.BatchGenerator()

# iterate through entire dataset
for mb_x, mb_y in batchgen.get_batch(X, y, batch_size=batch_size): pass

# test batch shuffling
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=True)
mb_x, mb_y = batchIterator.next()
if np.all(mb_x == X[:batch_size]) and np.all(mb_y == y[:batch_size]): 
    raise Exception("Error correctly shuffling generation of batches")

# get static non-shuffled batch as reference for augmentations
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False)
mb_x, mb_y = batchIterator.next()
if not np.all(mb_x == X[:batch_size]) and not np.all(mb_y == y[:batch_size]): 
    raise Exception("Error correctly generating batch with no shuffle")

# set up batch generator with augmentation parameters
augmentation_params = dict(
        rotation = 15,
        zoom = (1.5, 1.5), # zoom (x, y) = (col, row)
        shear = 7,
        translation = (5, -5),
        flip_lr = True,
        )
batchgen.set_aug_params(input_shape, aug_params=augmentation_params)

# get augmented batch
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False)
mb_x_aug, mb_y_aug = batchIterator.next()
if not np.all(mb_y == mb_y_aug): 
    raise Exception("Setting augmentations created error in generation of truth")

# set up batch generator with random augmentations
rng_augmentation_params = dict(
    zoom_range = (1/1.5, 1.5),
    rotation_range = (-15, 15),
    shear_range = (-7, 7),
    translation_range = (-5, 5),
    do_flip_lr = True,
    allow_stretch = True,
)
batchgen.aug_tf = None # unset static augmentation transforms so it doesnt do both.
batchgen.set_rng_aug_params(input_shape, rng_aug_params=rng_augmentation_params)

# get randomly augmented batch
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False)
mb_x_rng, mb_y_rng = batchIterator.next()
if not np.all(mb_y == mb_y_rng): 
    raise Exception("Setting rng augmentations created error in generation of truth")
if np.all(mb_x_rng == mb_x_aug) or np.all(mb_x_rng == mb_x):
    raise Exception("Random augmentations not correctly done")

# plot 3 random images from static batch_set, static augment set, & rng augment set
rng_idxs = np.arange(batch_size)
rng_idxs = np.random.choice(rng_idxs, size=3, replace=False)
plt.figure(1); plt.clf()
fig, axes = plt.subplots(3, 3, num=1)
for it, ax in enumerate(axes[:, 0]):
    img = mb_x[rng_idxs[it]].astype(np.uint8)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if it == 0:
        ax.set_title("Original Images")
        
for it, ax in enumerate(axes[:, 1]):
    img = mb_x_aug[rng_idxs[it]].astype(np.uint8)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if it == 0:
        ax.set_title("Static Augmentations")
        
for it, ax in enumerate(axes[:, 2]):
    img = mb_x_rng[rng_idxs[it]].astype(np.uint8)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    if it == 0:
        ax.set_title("Random Augmentations")