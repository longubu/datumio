# COPYRIGHT
# ---------
# All contributions by Long Van Ho:
# Copyright (c) 2015 Long Van Ho
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2015, the respective contributors.
# All rights reserved.
#
# LICENSE
# ---------
# The MIT License (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN
# ==============================================================================

"""
datumio.datagen
---------
Data generators used for loading datasets in mini-batches.

Generators
---------
BatchGenerator:
    Mini-batch generator for datasets that can be loaded entirely into memory
DataGenerator:
    Mini-batch generator for datasets that can't entirely fit into memory.

Idea taken from Keras's implementation: Keras.preprocessing.image.py
"""

import numpy as np

# Local imports
import transforms as dtf
import buffering as dtb
import data_utils as dtu

class InstantiateError(Exception):
    """Raise error if base class was not peroperly instantiated by its
    super class"""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


class BaseGenerator(object):
    """Abstract base generator class.

    See `BatchGenerator` for parameter descriptions.
    """
    def __init__(self, X, y=None,
                 aug_params=None, rng_aug_params=None,
                 dataset_zmuv=False, dataset_axis=None,
                 batch_zmuv=False, batch_axis=None,
                 sample_zmuv=False, sample_axis=None):

        # set local properties
        self.aug_params = aug_params
        self.rng_aug_params = rng_aug_params
        self.dataset_zmuv = dataset_zmuv
        self.dataset_axis = dataset_axis
        self.batch_zmuv = batch_zmuv
        self.batch_axis = batch_axis
        self.sample_zmuv = sample_zmuv
        self.sample_axis = sample_axis

        # set input data & truth
        self.X = np.array(X)
        if y is not None:
            self.y = np.array(y)
        else:
            self.y = None

        # addn' default used to imply processing actions: see `set_actions`
        self.mean = None
        self.std = None
        self.tf = None
        self.greyscale = False

    @property
    def input_shape(self):
        """Get shape of input data based on first data in X"""
        return np.shape(self.data_loader(self.X[0], **self.dl_kwargs))

    def resample_dataset(self, arr, weights, sample_fraction=1.0,
                         rng_seed=None):
        """Resamples unbalanaced dataset according to labels in `arr` and
        `weights`. This modifies self.X and self.y.

        Parameters
        -------
        arr: array-like, shape = [n_samples] or [n_samples, outputs]
            Array of labels/class/group to balance.

        class_weight: dict, list of dicts, "balanced", or None, optional
            Weights associated with `arr` in the form ``{label: weight}``,
            where the keys `label` are unique values present in arr and
            weights are the percentage of which to sample. If not given, all
            classes are set to weights of 1. For multi-output problems, a list
            of dicts can be provided in the same order as the columns of y.\n

            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the data.\n

            For multi-output, the weights of each column of y will be
            multiplied.

        sample_fraction: float, default=1.0
            Fraction of len(arr) to return when resampling dataset.
            For example, if `sample_fraction=2.0`, will return arr of
            ``len(2*len(arr))``.

        rng_seed: int, default=None
            Seed to random state that uses np.choice to select idxs of samples.

        Returns
        -------
        idxs: ndarray, shape = (sample_fraction * len(arr))
            Resampled indices of arr according to weights.
            Will apply: self.X = self.X[idxs]
                        self.y = self.y[idxs]
        """
        idxs = dtu.resample_data(arr, weights, sample_fraction=sample_fraction,
                                 rng_seed=rng_seed)
        self.X = self.X[idxs]
        self.y = self.y[idxs]
        print("[datumio] resampling dataset to len(X) = %s" % len(self.X))
        print("[datumio] ... Note: This is sampled with replacement. An "
              "epoch no longer means a pass through the dataset.")
        return idxs

    def set_actions(self):
        """Set generator processing stream actions: dataset_zmuv, static aug

        All other actions are implied from initialization:
            - batch_zmuv
            - sample_zmuv
            - rng_aug_params
        """
        input_shape = self.input_shape[:2]

        # compute mean & std of dataset
        if self.dataset_zmuv:
            self.mean, self.std = self.compute_dataset_moments()

        # pre-build static augmentation transform
        if self.aug_params is not None:
            self.warp_kwargs = self.aug_params.pop('warp_kwargs', None)
            self.tf = dtf.build_augmentation_transform(input_shape,
                                                       **self.aug_params)
            self.output_shape = self.aug_params.pop('output_shape', None)

        # if input is greyscale. used for `chw_order` in `get_batch`
        if len(self.input_shape) == 2:
            self.greyscale = True

    def standardize(self, x):
        """Applies generator processing to a loaded data x:
            - dataset zmuv
            - sample zmuv
            - static augmentation
            - rng augmentation

        Batch augmentations are done after loading a batch
        """
        # do dataset zmuv
        if (self.mean is not None) and (self.std is not None):
            x = x - self.mean
            x = x / (self.std + 1e-12)

        # do sample zmuv
        if self.sample_zmuv:
            x = x - np.mean(x, axis=self.sample_axis)
            x = x / (np.std(x, axis=self.sample_axis) + 1e-12)

        # apply static augmentations
        if self.tf is not None:
            x = dtf.transform_image(x, output_shape=self.output_shape,
                                    tf=self.tf, warp_kwargs=self.warp_kwargs)

        # apply random augmentations
        if self.rng_aug_params is not None:
            x = dtf.perturb_image(x, **self.rng_aug_params)

        return x

    def get_batch(self, batch_size=32, shuffle=False, rng_seed=None,
                  buffer_size=2, dtype=np.float32, chw_order=False):
        """Buffered generator. Returns minibatches of dataset (X, y) w/
        real-time augmentations applied on-the-fly. If y is not provided,
        get_batch will only return minibatches of X.

        Parameters
        ---------
        batch_size: int, default=32
            Size of minibatches to extract from X. If X % batch_size != 0,
            then the last batch returned the remainder, X % batch_size.

        shuffle: bool, default=False
            Whether to shuffle X and y before generating minibatches.

        rng_seed: int, default=None
            Seed to random state that shuffles X,y (if `shuffle=true`).

        buffer_size: int, default=2
            Size of to load in the buffer with each call.

        dtype: np.dtype, default=np.dtype32
            Data type of minibatch to be returned.

        chw_order: bool, default=False
            Return shape of minibatch. If False, minibatch returns will be of
            shape (batch_size, height, width, channel). If True, minibatches
            will be return of shape (batch_size, channel, height, width)

        Yield
        ---------
        ret: tuple OR ndarray
            If y is None (supplied at initialization of generator), returns
            minibatch of X with shape depending on `chw_order`.

            If y is initialized, returns tuple (mb_x, mb_y), where mb_x
            is minibatch of X and mb_y is minibatch of y wit shape
            depending on `chw_order`.
        """
        ndata = len(self.X)

        # set randomstate for shuffling data, if supplied
        if rng_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(seed=rng_seed)

        # index to iterate through X, y
        idxs = range(ndata)
        if shuffle:
            rng.shuffle(idxs)

        # set up generator with buffer
        def gen_batch():
            # generate batches
            nb_batch = int(np.ceil(float(ndata)/batch_size))
            for b in range(nb_batch):
                # determine batch size. all should equal bsize except the
                # last batch, when len(X) % bsize != 0.
                batch_end = (b + 1) * batch_size
                if batch_end > ndata:
                    nb_samples = ndata - b * batch_size
                else:
                    nb_samples = batch_size

                # get a minibatch
                bX = []
                for i in xrange(nb_samples):
                    idx = idxs[(b * batch_size) + i]
                    x = np.array(
                        self.data_loader(self.X[idx], **self.dl_kwargs),
                        dtype=np.float32)

                    # apply actions: zmuv, static_aug, rng_aug, etc.
                    x = self.standardize(x)
                    bX.append(x)
                bX = np.array(bX, dtype=dtype)

                # do batch zmuv
                if self.batch_zmuv:
                    bX = bX - bX.mean(axis=self.batch_axis)
                    bX = bX / (bX.std(axis=self.batch_axis) + 1e-12)

                if chw_order:
                    if self.greyscale:
                        bX = np.expand_dims(bX, 3)
                    bX = bX.transpose(0, 3, 1, 2)

                if self.y is not None:
                    bslice = idxs[b * batch_size: b * batch_size + nb_samples]
                    yield bX, self.y[bslice]
                else:
                    yield bX

        return dtb.buffered_gen_threaded(gen_batch(), buffer_size=buffer_size)

    # --- functions that need to be defined in parent class --- #

    def compute_dataset_moments(self):
        """Computes mean, std of dataset.

        Returns
        ------
        (mean, std): tuple
            Of floats or ndarrays (if computation is done on axis != None)
        """
        raise InstantiateError("compute_dataset_moments not instantiated")

    def set_data_loader(self):
        """Sets data_loader for generator. For BatchGenerator, returns itself.
        For DataGenerator, user defines a function that loads in their data.

        Function should set the following properties
        ------
        self.data_loader: func
            Python function that loads objects of X to return data.

        self.dl_kwargs: dict
            Keyword arguments to `self.data_loader`. If no kwargs are required,
            set dl_kwargs = {}

        Returns
        ------
        None
        """
        raise InstantiateError("set_data_loader not instantiated")


class BatchGenerator(BaseGenerator):
    """Batch generator with realtime data augmentation.
    Requires loading the dataset onto memory beforehand.

    Parameters
    ------
    X: iterable, ndarray
        Dataset to generate batch from.
        X.shape must be (dataset_length, height, width, channels)

    y: iterable, ndarray, default=None
        Corresponding labels to dataset. If label is None, get_batch will
        only return minibatches of X. y.shape = (data, ) or
        (data, one-hot-encoded)

    dataset_zmuv: bool, default=False
        Subtracts mean and divides by std of entire dataset on each sample x.

    dataset_axis: None or int or tuple of ints, optional
        Axis or axes along which dataset mean,std are computed. See `np.mean`
        axis option. If dataset_zmuv=False, this does not matter.

    batch_zmuv: bool, default=False
        Subtracts mean and divides by std within each minibatch load on each
        sample x.

    batch_axis: None or int or tuple of ints, optional
        Axis or axes along which batch mean,std are computed. See `np.mean`
        axis option. If batch_zmuv=False, this does not matter.

    sample_zmuv: bool, default=False
        Subtracts mean and divides by std of x on itself.

    sample_axis: None or int or tuple of ints, optional
        Axis or axes along which sample_mean,std are computed. See `np.mean`
        axis option. If sample_zmuv=False, this does not matter.

    Examples
    ------
    X # array of loaded images of shape (100, 32, 32, 3)

    BatchGen = BatchGenerator(X, shuffle=True, dataset_zmuv=True)
    for mb_x in BatchGen.get_batch():
        # do something with mb_x, a minibatch load from X with on-the-fly
        # augmentations/zmuv
        pass

    See `examples/cifar10_cnn_batchgen.py` for more thorough example.
    """
    def __init__(self, X, y=None,
                 aug_params=None, rng_aug_params=None,
                 dataset_zmuv=False, dataset_axis=None,
                 batch_zmuv=False, batch_axis=None,
                 sample_zmuv=False, sample_axis=None):

        kwargs = locals()
        if 'self' in kwargs:
            kwargs.pop('self')

        # set data loader -- this just returns itself. provided for generality
        self.set_data_loader()

        super(BatchGenerator, self).__init__(**kwargs)
        self.set_actions()

    def compute_dataset_moments(self):
        """Compute mean and std of entire dataset"""
        mean = self.X.astype(np.float32).mean(self.dataset_axis)
        std = (self.X.astype(np.float32) - mean).std(self.dataset_axis)
        return (mean, std)

    def set_data_loader(self):
        """Set data_loader for generator. This is handled for generality.
        For BatchGenerator, this returns itself"""
        def data_loader(x):
            return x

        self.data_loader = data_loader
        self.dl_kwargs = {}


class DataGenerator(BaseGenerator):
    """Batch generator with realtime data augmentation. Data is loaded
    and augmented on-the-fly.

    Parameters
    ------
    X: iterable, ndarray
        Path to each data file within a dataset. Each sample of X is loaded
        using `data_loader`.
        X.shape must be (dataset_length, )

    y: iterable, ndarray, default=None
        Corresponding labels to dataset. If label is None, get_batch will
        only return minibatches of X. y.shape = (data, ) or
        (data, one-hot-encoded)

    data_loader: func, default=dtu.img_loader
        Function used for loading each sample of `X`. Default loader,
        `dtu.img_loader` is a generic function that loads standard img
        files (png, jpg, tifs, etc) and npy arrays (in the shape of an image)

    dl_kwargs: dict, default=None
        Keyword arguments to pass to `data_loader` when loading samples of X.
        If None, no kwargs will passed to data_loader.

    dataset_zmuv: bool, default=False
        Subtracts mean and divides by std of entire dataset on each sample x.

    dataset_axis: None or int or tuple of ints, optional
        Axis or axes along which dataset mean,std are computed. See `np.mean`
        axis option. If dataset_zmuv=False, this does not matter.

    dataset_zmuv_bsize: int, default=32
        If dataset_zmuv == True, this is the size of minibatches to load per
        yield to compute the dataset mean and std for zero-mean unit variance.

    batch_zmuv: bool, default=False
        Subtracts mean and divides by std within each minibatch load on each
        sample x.

    batch_axis: None or int or tuple of ints, optional
        Axis or axes along which batch mean,std are computed. See `np.mean`
        axis option. If batch_zmuv=False, this does not matter.

    sample_zmuv: bool, default=False
        Subtracts mean and divides by std of x on itself.

    sample_axis: None or int or tuple of ints, optional
        Axis or axes along which sample_mean,std are computed. See `np.mean`
        axis option. If sample_zmuv=False, this does not matter.

    Examples
    ------
    X # path images of shape (100,)

    DataGen = DataGenerator(X, shuffle=True, dataset_zmuv=True)
    for mb_x in DataGen.get_batch():
        # do something with mb_x, a minibatch load from X with on-the-fly
        # augmentations/zmuv
        pass

    See `examples/cifar10_cnn_datagen.py` for more thorough example.
    """
    def __init__(self, X, y=None, data_loader=dtu.img_loader, dl_kwargs=None,
                 aug_params=None, rng_aug_params=None,
                 dataset_zmuv=False, dataset_axis=None, dataset_zmuv_bsize=32,
                 batch_zmuv=False, batch_axis=None,
                 sample_zmuv=False, sample_axis=None):

        kwargs = locals()
        if 'self' in kwargs:
            kwargs.pop('self')

        # init variables that aren't standard to base class
        self.dataset_zmuv_bsize = kwargs.pop('dataset_zmuv_bsize')

        # set data loader
        data_loader = kwargs.pop('data_loader')
        dl_kwargs = kwargs.pop('dl_kwargs')
        self.set_data_loader(data_loader, dl_kwargs)

        super(DataGenerator, self).__init__(**kwargs)
        self.set_actions()

    def compute_dataset_moments(self):
        """Compute mean and std of entire dataset by taking the average
        of the mean and std of loaded minibatches. Each minibatch (whose
        size is equal to initialized batch_size) is loaded using
        get_batch, with augmentations and zmuv computations turned off"""
        # store states so we can use get_batch w/o addn actions
        tf = self.tf
        rng_aug_params = self.rng_aug_params
        batch_zmuv = self.batch_zmuv
        sample_zmuv = self.sample_zmuv

        self.tf = None
        self.rng_aug_params = None
        self.batch_zmuv = False
        self.sample_zmuv = False

        # compute mean/std in batches
        batches_mean = []
        batches_std = []
        for ret in self.get_batch(batch_size=self.dataset_zmuv_bsize):
            if self.y is None:
                mb_x = ret
            else:
                mb_x = ret[0]

            mean = mb_x.mean(axis=self.dataset_axis)
            std = (mb_x - mean).std(axis=self.dataset_axis)

            batches_mean.append(mean)
            batches_std.append(std)

        # reset state
        self.tf = tf
        self.rng_aug_params = rng_aug_params
        self.batch_zmuv = batch_zmuv
        self.sample_zmuv = sample_zmuv

        return (np.mean(batches_mean, axis=0), np.mean(batches_std, axis=0))

    def set_data_loader(self, data_loader, dl_kwargs):
        """Set data_loader for generator to load each sample of X into
        data that can be augmented"""
        self.data_loader = data_loader
        if dl_kwargs is None:
            self.dl_kwargs = {}
        else:
            self.dl_kwargs = dl_kwargs
