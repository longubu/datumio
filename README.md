# datumio
Datumio is a data loader with real-time augmentation, specifically tailored for inputs into image-based deep learning models (Convolutional Neural Networks).

Datumio is an aggregation of other existing open source projects (specifically kaggle-plankton, kaggle-galaxy-zoo, keras-preprocessing-image) with hopes that it is simple and general enough to be used with any framework, from `keras` to `caffe`.

Datumio purpose:
- load datasets that may not entirely fit into memory
- fast, (random and static) augmentation of dataset upon loading
- agnostic to deeplearning framework
- parallel processing of GPU and CPU - data is constantly streaming via the CPU as the GPU does computation on the previous minibatch.

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
