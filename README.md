# datumio
Real-time augmentation of data for inputs into deep learning models.

## Description
In practice, entire datasets do not fit into memory. Furthermore, we often want to augment our datasets to
robustify our models and ultimately increase the prediction accuracy of the model during its realtime deployments.

This repo aggregates existing open source codes that I found (...), with hopes that it is simple and general
enough to supply any model with fast realtime augmentations of their inputs.

## _fluff_
Neural network models such as CNNs require large amounts of labeled data. In practice, this data lake is usually 
small, leading to issues such as overfitting. Consequenetly, methods such as "data augmentation" aim to expand 
the dataset by perturbing the existing data, usually through affine transformations such as pixel translations, 
rotations, and zooms. This creates other "views" of the same object for the CNN to learn more robust features,
mediating the problems of overfitting.

