"""
datumio version of keras/examples/cifar10_cnn.py
Note: This also does resampling of the dataset. This is useful for datasets
with unbalanced labels.

Copied and verifeid to work with keras.__version__ == 0.2.0

Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn_datagen.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50
epochs. (it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might
prevent you from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
"""

from __future__ import print_function
import datumio.datagen as dtd
from keras.datasets import cifar10
# from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range

import numpy as np
import os
from PIL import Image
import shutil

batch_size = 32
nb_classes = 10
nb_epoch = 20
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# datumio: datagen loads data from disk. Save X_train/X_test to disk
if not os.path.exists('cifar-10-train'): os.mkdir('cifar-10-train')
X_train_paths = []
for i, x in enumerate(X_train):
    path = os.path.abspath(os.path.join('cifar-10-train', '%06d.png' % (i+1)))
    X_train_paths.append(path)
    Image.fromarray(x.transpose(1, 2, 0)).save(path)

if not os.path.exists('cifar-10-test'): os.mkdir('cifar-10-test')
X_test_paths = []
for i, x in enumerate(X_test):
    path = os.path.abspath(os.path.join('cifar-10-test', '%06d.png' % (i+1)))
    X_test_paths.append(path)
    Image.fromarray(x.transpose(1, 2, 0)).save(path)

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

if not data_augmentation:
    print('Not using data augmentation or normalization')
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score:', score)

else:
    print('Using real time data augmentation')

    rng_aug_params = {'rotation_range': (-20, 20),
                      'translation_range': (-4, 4),
                      'do_flip_lr': True}

    # DataGenerator takes X = path-to-data-files instead of actually loaded X.
    X_train = np.array(X_train_paths)
    X_test = np.array(X_test_paths)

    # iterate through dataset. Note we have to redefine the generators
    # each we "resample" the dataset.
    errors = []
    for e in range(nb_epoch):

        datagen = dtd.DataGenerator(X_train, y=Y_train,
                                    rng_aug_params=rng_aug_params,
                                    dataset_zmuv=True, dataset_axis=0)

        # resample the dataset for balanced labels
        datagen.resample_dataset(Y_train, 'balanced')
        testgen = dtd.DataGenerator(X_test, y=Y_test,
                                    dataset_zmuv=True, dataset_axis=0)

        # make sure mean, std subtracted from test batches are same as ones
        # from train batches.
        testgen.mean = datagen.mean
        testgen.std = datagen.std

        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print('Training...')
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.get_batch(
                                    batch_size=batch_size, shuffle=True,
                                    chw_order=True):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('train loss', loss[0])])

        print('Testing...')
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        scores = []
        for X_batch, Y_batch in testgen.get_batch(
                                    batch_size=batch_size, chw_order=True):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[('test loss', score[0])])
            scores.append(score)

        errors.append(np.mean(scores))

    print(errors)

# datumio: clean up folders
shutil.rmtree('./cifar-10-train')
shutil.rmtree('./cifar-10-test')
