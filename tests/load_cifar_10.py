"""
Utility function to load cifar-10 data
"""
import numpy as np
import pandas as pd
from PIL import Image
import os

def load_cifar10_data():
    dataDir = 'test_data/cifar-10/'
    labelPath = 'test_data/cifar-10/labels.csv'
    labelDF = pd.read_csv(labelPath)
    uid_labels = labelDF.values
    X = []
    for uid, label in uid_labels:
        img = Image.open(os.path.join(dataDir, uid))
        X += [np.array(img)]
    X = np.array(X, dtype = np.uint8)
    y = np.array(uid_labels[:, 1], dtype = int)
    return X, y