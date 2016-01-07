from keras.datasets.data_utils import get_file
from keras.datasets.cifar import load_batch
import numpy as np
import os
import pandas as pd
from PIL import Image


def get_and_save_data(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dirname = "cifar-10-batches-py"
    origin = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    path = get_file(dirname, origin=origin, untar=True)

    nb_train_samples = 10000

    X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
    y_train = np.zeros((nb_train_samples,), dtype="uint8")

    for i in range(1, 2):  # only downlaod one set instead of 5
        fpath = os.path.join(path, 'data_batch_' + str(i))
        data, labels = load_batch(fpath)
        X_train[(i-1)*10000:i*10000, :, :, :] = data
        y_train[(i-1)*10000:i*10000] = labels

    # reshape into standard shape: (10000, 32, 32, 3)
    X_train = X_train.transpose(0, 2, 3, 1)

    paths = []
    for i, x in enumerate(X_train):
        path = os.path.abspath(os.path.join(save_dir, '%05d.png' % i))
        paths.append(path)
        Image.fromarray(x).save(path)

    data_info = pd.DataFrame(zip(paths, y_train), columns=['uid', 'label'])
    data_info.to_csv(save_dir, 'labels.csv', index=False)
