from __future__ import absolute_import
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from six.moves import range
from datumio.datagen import BatchGenerator

import numpy as np

'''
    Train a (fairly simple) deep CNN on the CIFAR10 small images dataset.

    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

    It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
    (it's still underfitting at that point, though).

    Note: the data was pickled with Python 2, and some encoding issues might prevent you
    from loading it in Python 3. You might have to load it in Python 2,
    save it in a different format, load it in Python 3 and repickle it.
'''
import time

batch_size = 32
nb_classes = 10
nb_epoch = 1

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

print("Using real time data augmentation")

#==============================================================================
# preprocessing using datumio
#==============================================================================
batchgen = BatchGenerator()
batchgen.set_global_zmuv(X_train.transpose(0,2,3,1), axis=0)
rng_aug_params = {'rotation_range': (-20, 20),
                  'translation_range': (-4, 4),
                  'do_flip_lr': True,}
                  
batchgen.set_rng_aug_params(rng_aug_params)
times2 = []

print('starting data io timing for datumio')
for e in range(nb_epoch):
    progbar = generic_utils.Progbar(X_train.shape[0])
    start_time = time.time()
    for d_X_batch, d_Y_batch in batchgen.get_batch(X_train.transpose(0,2,3,1), labels=y_train,
                                               batch_size=batch_size,
                                               ret_opts={'chw_order':True},
                                               shuffle=False):
        pass
    
    delta_time = time.time() - start_time
    times2.append(delta_time)
    break

print('... datumio data IO per epoch took <%s>'%np.mean(times2))

#==============================================================================
#  preprocessing using keras
#==============================================================================
datagen = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

times = []
print('starting data io timing for Keras data IO')
for e in range(nb_epoch):
    # batch train with realtime data augmentation
    progbar = generic_utils.Progbar(X_train.shape[0])
    start_time = time.time()
    for X_batch, Y_batch in datagen.flow(X_train, y_train, shuffle=False,
                                         batch_size=batch_size): 
        pass

    delta_time = time.time() - start_time
    times.append(delta_time)
    break

print('... Keras data IO per epoch took <%s>'%np.mean(times))
    
#==============================================================================
# compare visually the rng augmentation from both 
#==============================================================================
import matplotlib.pyplot as plt

plt.figure(1, figsize=(12,10)); plt.clf()
fig, axes = plt.subplots(4, 3, num=1)
for idx, ax in enumerate(axes[:, 0]):
    if idx ==0:
        ax.set_title("Original %s"%(y_train[-idx]))
    else:
        ax.set_title('%s'%y_train[-idx])
    ax.imshow(X_train[-idx].transpose(1,2,0))
    plt.axis('off')

for idx, ax in enumerate(axes[:, 1]):
    if idx == 0:
        ax.set_title("Datumio %s"%(d_Y_batch[-idx]))
    else:
        ax.set_title('%s'%(Y_batch[-idx]))
    ax.imshow(d_X_batch[-idx].transpose(1,2,0))
    plt.axis('off')

for idx, ax in enumerate(axes[:, 2]):
    if idx == 0:
        ax.set_title("Keras: %s"%(Y_batch[-idx]))
    else:
        ax.set_title('%s'%(Y_batch[-idx]))
    ax.imshow(X_batch[-idx].transpose(1,2,0))
    plt.axis('off')

plt.tight_layout()    



















