"""
Test datumio.datagen.BatchGenerator
"""
import numpy as np
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

import datumio.datagen as dtd

#==============================================================================
# load data & labels
#==============================================================================
dataDir = 'test_data/cifar-10/'
labelPath = 'test_data/cifar-10/labels.csv'
labelDF = pd.read_csv(labelPath)
uid_labels = labelDF.values
X = np.array([np.array(Image.open(os.path.join(dataDir, uid))) for uid in uid_labels[:, 0]], dtype=np.uint8)
y = np.array(uid_labels[:, 1], dtype=int)

#==============================================================================
# test batch generator with zmuv & shuffle
#==============================================================================
# set up batch generator
batch_size = 32
batchgen = dtd.BatchGenerator()

# test if minibatches equal to the minibatches we extract manually
for idx, (mb_x, mb_y) in enumerate(batchgen.get_batch(X, y, 
                                    batch_size=batch_size, shuffle=False)): 
    if ~(np.all(mb_x == X[idx*batch_size: (idx+1)*batch_size])):
        raise Exception("Not all minibatches X match correctly from dataset")
    if ~(np.all(mb_y == y[idx*batch_size: (idx+1)*batch_size])):
        raise Exception("Not all minibatches y match correctly from dataset")        

# test batch shuffling, assuming the top runs then just test if opposite
for idx, (mb_x, mb_y) in enumerate(batchgen.get_batch(X, y, 
                                    batch_size=batch_size, shuffle=True)): 
    if (np.all(mb_x == X[idx*batch_size: (idx+1)*batch_size])):
        raise Exception("Minibatches in X are not correctly shuffled")
    if (np.all(mb_y == y[idx*batch_size: (idx+1)*batch_size])):
        raise Exception("Minibatches in y are not correctly shuffled")  

# get a batch of non-shuffled data so we can visually
# compare with zmuv & augmentations on dataset
batchgen = dtd.BatchGenerator()
mb_x_og, mb_y_og = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()

# test global zero-mean unit-variance
# compute manually
mean = X.mean(axis=0)
std = X.std(axis=0)
zmuv_mb_x = mb_x_og - mean
zmuv_mb_x = zmuv_mb_x / std

# using batchgen 
batchgen = dtd.BatchGenerator()
batchgen.set_global_zmuv(X, axis=0)
mb_x, mb_y = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()

# compare global zmuv. Due to floating point precision, 
# lets cehck only the first image w/ str 4pt precision
zmuv_mb_x_img0 = np.array(['%0.4f'%x for x in zmuv_mb_x[0].flatten()])
mb_x_img0 = np.array(['%0.4f'%x for x in mb_x[0].flatten()])
if not np.all(mb_x_img0 == zmuv_mb_x_img0):
    raise Exception("Error correctly generating batch with global zmuv")

# test samplewise zmuv
mb_x_manual_samplewise = []
for x in mb_x_og.copy():
    x -= x.mean(axis=0)
    x /= x.std(axis=0)
    mb_x_manual_samplewise.append(x)
mb_x_manual_samplewise = np.array(mb_x_manual_samplewise)

batchgen = dtd.BatchGenerator()
batchgen.set_samplewise_zmuv(axis=0)
mb_x, mb_y = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()

# compare samplewise zmuv.
if not np.all(mb_x == mb_x_manual_samplewise):
    raise Exception("Error correctly generating batch with samplewise zmuv")
    
#==============================================================================
# test batch generator with augmentations & compare visually through figures
#==============================================================================
# set up batch generator with augmentation parameters
augmentation_params = dict(
        rotation = 15,
        zoom = (1.5, 1.5), # zoom (x, y) = (col, row)
        shear = 7,
        translation = (5, -5),
        flip_lr = True,
        )
batchgen = dtd.BatchGenerator(do_static_aug=True)
input_shape = (32, 32)
batchgen.set_static_aug_params(input_shape, aug_params=augmentation_params)

# get augmented batch
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False)
mb_x_aug, mb_y_aug = batchIterator.next()
if not np.all(mb_y_og == mb_y_aug): 
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
batchgen.do_static_aug=False # unset static aug
batchgen.set_rng_aug_params(rng_aug_params=rng_augmentation_params)

# get randomly augmented batch
batchIterator = batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False)
mb_x_rng, mb_y_rng = batchIterator.next()

if not np.all(mb_y_og == mb_y_rng): 
    raise Exception("Setting rng augmentations created error in generation of truth")
if np.all(mb_x_rng == mb_x_aug) or np.all(mb_x_rng == mb_x_og):
    raise Exception("Random augmentations not correctly done")

# plot 3 random images from static batch_set, static augment set, & rng augment set
rng_idxs = np.arange(batch_size)
rng_idxs = np.random.choice(rng_idxs, size=3, replace=False)
plt.figure(1); plt.clf()
fig, axes = plt.subplots(3, 3, num=1)
for it, ax in enumerate(axes[:, 0]):
    img = mb_x_og[rng_idxs[it]].astype(np.uint8)
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
        
#==============================================================================
# test error checks
#==============================================================================
# check when user creates do_* but not using set_* subsequently
print("---- Raise warning about not setting global zm ----")
batchgen = dtd.BatchGenerator(do_global_zm=True)
try:
    batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()
except Exception,e:
    print(e)
    print

print("---- Raise warning about not setting global uv ----")
batchgen = dtd.BatchGenerator(do_global_uv=True)
try:
    batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()
except Exception,e:
    print(e)
    print
    
print("---- Raise warning about not setting static augmentation ----")
batchgen = dtd.BatchGenerator(do_static_aug=True)
try:
    batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()
except Exception,e:
    print(e)
    print
    
print("---- Raise warning about not setting random augmentation ----")
batchgen = dtd.BatchGenerator(do_rng_aug=True)
try:
    batchgen.get_batch(X, y, batch_size=batch_size, shuffle=False).next()
except Exception,e:
    print(e)
    print   

# test setting both random & static augmentations warning
print(" ---- Raise warning about setting both static & rng augmentatinos ----")
batchgen = dtd.BatchGenerator(do_static_aug=True, do_rng_aug=True)
batchgen.set_static_aug_params(input_shape, aug_params=augmentation_params)
batchgen.set_rng_aug_params(rng_aug_params=rng_augmentation_params)

print("#===========================================================================")
print("# Compare Figure (1) for transforms. Compare warnings with what was expected")
print("# Everything else looks good")
print("#===========================================================================")
