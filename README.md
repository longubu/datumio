# datumio
Real-time augmentation of data for inputs into deep learning models.

## Description
In practice, datasets do not always fit entirely into the memory. Furthermore, we often want to artificially augment our datasets to
robustify our models and ultimately increase the prediction accuracy of the model during its realtime deployments.

This repo aggregates existing open source codes that I found (kaggle-plankton, kaggle-galaxy-zoo, keras-preprocessing-image), with hopes that it is simple and general
enough to supply any model with fast realtime augmentations of their inputs.

# Downsample branch
Finished this branch is able to merge. However, after some testing, found that it is actually slower than just downsampling the image and then applying the augmentations.
