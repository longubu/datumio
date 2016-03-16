# datumio
Datumio is a data loader with real-time augmentation, specifically tailored for inputs into image-based deep learning models (Convolutional Neural Networks).

Datumio is an aggregation of other existing open source projects (specifically kaggle-plankton, kaggle-galaxy-zoo, keras-preprocessing-image) with hopes that it is simple and general enough to be used with any framework, from `keras` to `caffe`.

Datumio purpose:
- load datasets that may not entirely fit into memory
- fast, (random and static) augmentation of dataset upon loading
- agnostic to deeplearning framework
- parallel processing of GPU and CPU - data is constantly streaming via the CPU as the GPU does computation on the previous minibatch.
- dataset re-sampling based on labels (or any externally supplied group). For example, this is useful for resampling datasets with unbalanced labels which can bias models to predicting the classes that appear more frequently.

# Installation
## Dependencies

- SciPy
- Numpy
- PIL
- scikit-image

If you don't already have these packages installed, it is
recommended to install via third party distribution such as anaconda. If you're using linux, it's recommended to install using the package manager -- else it may do a lengthy build process with many depedencies and may lead to a much slower configuration. 

## Build from source (does not include dependencies)

	git clone https://github.com/longubu/datumio.git
	cd datumio
	python setup.py install

# Usage
See `examples/*` for usage.

### keras
- [cifar10_cnn_batchgen.py](examples/keras/cifar10_cnn_batchgen.py): Example of using the batch generator with data that can fit entirely into memory.
- [cifar10_cnn_datagen.py](examples/keras/cifar10_cnn_datagen.py): Example of using the batch generator with data that must be loaded as needed and can not be loaded into memory.
- [cifar10_cnn_datagen_with_resampling.py](examples/keras/cifar10_cnn_datagen_with_resampling.py): Example of using the batch generator that samples batches in proportion to given labels (E.g. to sample labels uniformly from unbalanced datasets).
- [mnist_cnn_batchgen.py](examples/keras/mnist_cnn_batchgen.py): Example of batch generator on mnist dataset.

### tensorflow
- [mnist_cnn.py](examples/tensorflow/mnist_cnn.py): Example of using batch generator with MNIST to train a CNN using tensorflow

# TODO

- Create examples for other deep learning frameworks: [caffe, tensorflow, mxnet, etc]
- Extend to non-image data
- Create example for using it with single-file databases (lmdb, HDF5, etc)
