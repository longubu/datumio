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

TODO(Long): take into account greyscale images
TODO(Long): take into account order of operations

Idea taken from Keras's implementation: Keras.preprocessing.image.py
"""
import numpy as np
from PIL import Image

# Local imports
import transforms as dtf
import buffering as dtb


class BatchGenerator(object):
    """
    """
    def __init__(self, X, y=None,
                 batch_size=32, shuffle=False, rng_seed=None,
                 input_shape=None, aug_params=None, rng_aug_params=None,
                 dataset_zmuv=False, dataset_axis=None,
                 batch_zmuv=False, batch_axis=None,
                 sample_zmuv=False, sample_axis=None):

        # initial default states
        self.batch_size = batch_size
        self.aug_params = aug_params
        self.rng_aug_params = rng_aug_params
        self.dataset_zmuv = dataset_zmuv
        self.dataset_axis = dataset_axis
        self.batch_zmuv = batch_zmuv
        self.batch_axis = batch_axis
        self.sample_zmuv = sample_zmuv
        self.sample_axis = sample_axis

        # shuffle data based on rng, if supplied
        if rng_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(seed=rng_seed)

        # index to iterate through X, y
        idxs = range(len(X))
        if shuffle:
            rng.shuffle(idxs)

        # set input data & truth
        self.X = np.array(X)[idxs]
        if y is not None:
            self.y = np.array(y)[idxs]
        else:
            self.y = None

        if input_shape is not None:
            self.input_shape = input_shape
        else:
            self.input_shape = X.shape[1:3]

        self.mean = None
        self.std = None
        self.tf = None

        # set generator operations based on initializations
        self.__set_actions()

    def __set_actions(self):
        """Set generator processing stream"""

        # compute mean & std of dataset
        if self.dataset_zmuv:
            self.mean = self.X.astype(np.float32).mean(self.dataset_axis)
            self.std = (self.X.astype(np.float32) -
                        self.mean).std(self.dataset_axis)

        # pre-build static augmentation transform
        if self.aug_params is not None:
            self.warp_kwargs = self.aug_params.pop('warp_kwargs', None)
            self.tf = dtf.build_augmentation_transform(self.input_shape,
                                                       **self.aug_params)
            self.output_shape = self.aug_params.pop('output_shape', None)

    def __standardize(self, x):
        """Applies generator processor to 1 img/data"""
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

    def get_batch(self, buffer_size=2, dtype=np.float32, chw_order=False):
        """
        """
        bsize = self.batch_size

        # set up generator with buffer
        def gen_batch():
            # generate batches
            nb_batch = int(np.ceil(float(self.X.shape[0])/bsize))
            for b in range(nb_batch):
                # determine batch size. all should equal bsize except the
                # last batch, when len(X) % bsize != 0.
                batch_end = (b+1)*bsize
                if batch_end > self.X.shape[0]:
                    nb_samples = self.X.shape[0] - b*bsize
                else:
                    nb_samples = bsize

                # get a minibatch
                bX = []
                for i in xrange(nb_samples):
                    x = np.array(self.X[(b*bsize)+i], dtype=np.float32)

                    # apply actions: zmuv, static_aug, rng_aug, etc.
                    x = self.__standardize(x)
                    bX.append(x)
                bX = np.array(bX, dtype=dtype)

                # do batch zmuv
                if self.batch_zmuv:
                    bX = bX - bX.mean(axis=self.batch_axis)
                    bX = bX / (bX.std(axis=self.batch_axis) + 1e-12)

                if chw_order:
                    bX = bX.transpose(0, 3, 1, 2)

                if self.y is not None:
                    yield bX, self.y[b*bsize:b*bsize+nb_samples]
                else:
                    yield bX

        return dtb.buffered_gen_threaded(gen_batch(), buffer_size=buffer_size)


def img_loader(data_path):
    """ Generic function for loading images. Supports .npy & basic PIL.Image
    compatible extensions.

    Parameters
    ---------
    data_path: str
        Path to the image.

    Returns
    ---------
    img:
    """
    # get format of data, using the extension
    import os
    ext = os.path.basename(data_path).split(os.path.extsep)[1]

    if not os.path.exists(data_path):
        raise IOError("No such file: %s" % data_path)

    # load using numpy
    if ext == '.npy':
        img = np.load(data_path)

    # else default to PIL.Image supported extensions.
    # Loads most basic image formats.
    else:
        try:
            img = np.array(Image.open(data_path))
        except IOError:
            raise IOError("img_loader does not recognize file ext: %s"
                          % ext)
    return img


class DataGenerator(object):
    """
    """
    def __init__(self, X, y=None, data_loader=img_loader, dl_kwargs=None,
                 input_shape=None, aug_params=None, rng_aug_params=None,
                 dataset_zmuv=False, dataset_axis=None,
                 batch_zmuv=False, batch_axis=None,
                 sample_zmuv=False, sample_axis=None):

        # set initial states
        self.__dict__.update(locals())

        # make sure input dataset is ndarray for convenience
        self.X = np.array(X)
        if y is not None: self.y = np.array(y)

        # additional initial default states, some override init kwargs
        self.mean = None
        self.std = None
        self.tf = None
        if dl_kwargs is None: self.dl_kwargs = {}
        if input_shape is None: self.input_shape = self.__get_input_shape[:2]

        # set generator operations based on initializations
        self.__set_actions()

    @property
    def __get_input_shape(self):
        """Get input shape of image using first datapath of dataset"""
        return np.shape(self.data_loader(self.X[0], **self.dl_kwargs))

    def __set_actions(self):
        """Set generator processing stream"""

        # compute mean & std of dataset
        if self.dataset_zmuv:
            self.mean, self.std = self.__compute_dataset_moments(self.X)

        # pre-build static augmentation transform
        if self.aug_params is not None:
            self.warp_kwargs = self.aug_params.pop('warp_kwargs', None)
            self.tf = dtf.build_augmentation_transform(self.input_shape,
                                                       **self.aug_params)
            self.output_shape = self.aug_params.pop('output_shape', None)

    def __compute_dataset_moments(self, X):
        """Computes mean and std of entire dataset"""
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

        for ret in self.get_batch(batch_size=32):
            if self.y is None:
                mb_x = ret
            else:
                mb_x = ret[0]

            mean = mb_x.mean(axis=self.dataset_axis)
            std = (mb_x - mean).std(axis=self.dataset_axis)

            batches_mean.append(mean)
            batches_std.append(std)

        mean = np.mean(batches_mean, axis=0)
        std = np.mean(batches_std, axis=0)

        # reset state
        self.tf = tf
        self.rng_aug_params = rng_aug_params
        self.batch_zmuv = batch_zmuv
        self.sample_zmuv = sample_zmuv

        return (np.mean(batches_mean, axis=0), np.mean(batches_std, axis=0))

    def __standardize(self, x):
        """Applies generator processor to 1 img/data"""
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

    def get_batch(self, batch_size=32, shuffle=False, buffer_size=2,
                  rng_seed=None, dtype=np.float32, chw_order=False):
        """
        """
        if rng_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(seed=rng_seed)

        # index to iterate through X, y
        idxs = range(len(self.X))
        if shuffle: rng.shuffle(idxs)

        # set up generator with buffer
        def gen_batch():
            # generate batches
            nb_batch = int(np.ceil(float(self.X.shape[0])/batch_size))
            for b in range(nb_batch):
                # determine batch size. all should equal batch_size except the
                # last batch, when len(X) % batch_size != 0.
                batch_end = (b+1)*batch_size
                if batch_end > self.X.shape[0]:
                    nb_samples = self.X.shape[0] - b*batch_size
                else:
                    nb_samples = batch_size

                # get a minibatch
                batch_slice = idxs[b*batch_size:b*batch_size+nb_samples]
                bX = []
                for i in xrange(nb_samples):
                    # load based on data_path & set data_looader + kwargs
                    data_path = self.X[batch_slice[i]]
                    x = np.array(self.data_loader(data_path, **self.dl_kwargs),
                                 dtype=np.float32)

                    # apply actions: zmuv, static_aug, rng_aug, etc.
                    x = self.__standardize(x)
                    bX.append(x)
                bX = np.array(bX, dtype=dtype)

                # do batch zmuv
                if self.batch_zmuv:
                    bX = bX - bX.mean(axis=self.batch_axis)
                    bX = bX / (bX.std(axis=self.batch_axis) + 1e-12)

                if chw_order:
                    bX = bX.transpose(0, 3, 1, 2)

                if self.y is not None:
                    yield bX, self.y[batch_slice]
                else:
                    yield bX

        return dtb.buffered_gen_threaded(gen_batch(), buffer_size=buffer_size)


if __name__ == "__main__":
    import os
    import pandas as pd

    # set up data & truth
    data_dir = '../tests/test_data/cifar-10/'
    label_path = '../tests/test_data/cifar-10/labels.csv'
    label_df = pd.read_csv(label_path)
    uids = label_df['uid'].values
    y = label_df['label'].values

    data_paths = []
    for x in uids:
        data_paths.append(os.path.join(data_dir, x))

    datagen = DataGenerator(data_paths, y=y, dataset_zmuv=True)
