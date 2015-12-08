"""
Collection of data generator classes.

TODO: 
    - take into account greyscale images
"""
import numpy as np
from PIL import Image

import transforms as dtf
import buffering as dtb

class BatchGenerator(object):
    """
    Iterable batch fetcher with realtime data augmentation.
    Requires loading the dataset onto memory beforehand. Below, `mb`
    refers to minibatch.
    
    Parameters
    ---------
    do_global_zm: bool, optional
        Subtract mb by mean over the dataset. See `set_global_zmuv`
        
    do_global_uv: bool, optional
        Divide mb by std over the dataset. See `set_global_zmuv`
    
    do_samplewise_zm: bool, optional
        Subtract each sample of the mb by its mean. See `set_samplewise_zmuv`
    
    do_samplewise_uv: bool, optional
        Divide mb each sample of the mb by its std. See `set_samplewise_zmuv`
        
    do_static_aug: bool, optional
        Realtime augment of each mb with stationary augmentations: [crop, zoom, 
        rotation, shear, translation (x,y), flip_lr]. See `set_aug_params`.
        
    do_rng_aug: bool, optional
        Realtime augment of each mb with random augmentation [crop, zoom, 
        rotation, shear, translation (x,y), flip_lr]. See `set_rng_params`.
        
    Call
    ---------
    After initializing the BatchGenerator class, set each do_* procedure with 
    the associated set_* procedures. Then call by creating the generator:
    get_batch( ... ) and calling get_batch.next().
    """
    def __init__(self,
                 do_global_zm           = False,
                 do_global_uv           = False,
                 do_samplewise_zm       = False,
                 do_samplewise_uv       = False,
                 do_static_aug          = False,
                 do_rng_aug             = False,
                 ):
        
        self.__dict__.update(locals())
        self.mean                   = None
        self.std                    = None
        self.samplewise_zm_axis     = None
        self.samplewise_uv_axis     = None
        self.aug_tf                 = None
        self.rng_aug_params         = None
        
    def set_global_zmuv(self, X, axis=None):
        """
        Computes mean and std of dataset. Subtracts minibatch by mean
        and divides by std for global zero-mean unit-variance.
        
        Parameters
        ---------
        X: ndarray, shape = (data, height, width, channels)
            Dataset to generate batch from.
        
        axis: tuple, int, 0 (default)
            Axis to compute over. 0 will compute mean across batch with
            output shape (height, width, channels). (0,1,2) will compute
            mean across channels with output shape (3,). axis=None will
            take operation over flattened array.
            
        """
        zm_X = self._set_global_zm(X, axis=axis)
        uv_X = self._set_global_uv(zm_X, axis=axis)
        
    def set_samplewise_zmuv(self, axis=None):
        """
        Sets the axis over which to take the mean and std over each sample
        of the mb to substract and divide. All samples will have zero-mean 
        and unit-variance. See `set_global_umuv` for axis options.
        """
        self._set_samplewise_zm(axis=axis)
        self._set_samplewise_uv(axis=axis)
    
    def set_static_aug_params(self, input_shape, aug_params, warp_kwargs={}):
        """ Sets static augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params is not None and self.do_rng_aug: 
            print("[datagen:BatchGenerator] Warning: Static and Random augmentations are both set.")
         
        self.do_static_aug = True
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
        self.output_shape = aug_params.pop('output_shape', None)
        self.static_warp_kwargs = warp_kwargs
        
    def set_rng_aug_params(self, rng_aug_params):
        """ Sets random augmentation parameters to apply to each minibatch.
        See datumio.transforms.perturb_image."""
        if self.aug_tf is not None and self.do_static_aug:
            print("[datagen:BatchGenerator] Warning: Static and Random augmentations are both set.")
        
        self.do_rng_aug = True
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf

    def get_batch(self, X, labels=None, batch_size=32, shuffle=True, buffer_size=2,
                  rng=np.random, ret_opts={'dtype': np.float32, 'chw_order': False}):     
        """ 
        Iterable batch generator. Returns minibatches of the dataset (X, labels) with 
        realtime augmentaitons. If labels not provided, returns mb of (X,).
        Use get_batch.next() to fetch batches.
        
        Augmentation parameters and zmuv need to be set prior to running
        get_batch.
        
        Parameters
        ---------
        X: ndarray, shape = (data, height, width, channels)
            Dataset to generate batch from.
            
        labels: ndarray, shape = (data, ) or (data, one-hot-encoded), optional
            Corresponding labels to dataset. If label is None, get_batch will 
            only return minibatches of X.
        
        batch_size: int, optional
            Sizes of minibatches to extract from X. If X % batch_size != 0,
            then the last batch returned the remainder, X % batch_size.
        
        shuffle: bool, optional
            Whether to shuffle X and labels before generating minibatches
        
        buffer_size: int, optional
            Number of batches to generate in buffer such that subsequent calls
            return a preprocessed batch from the buffer and begin generating more.
            
        rng: np.random.RandomState, optional
            Randomstate to shuffle X (if true) for reproducibility.
        
        ret_opts: dict, optional
            Return options. 'dtype': is the datatype to return per minibatch,
            'chw_order': If true, will return the array in 
            (bsize, channels, height, width), else returns in
            (bsize, height, width, channels).
        """
        # parse ret_opts
        ret_opts = ret_opts.copy() # to not modify original object
        ret_dtype = ret_opts.pop('dtype', np.float32)
        ret_chw_order = ret_opts.pop('chw_order', False)
        
        # check if do_ procedures and params are correct
        self._check_do_params()
        
        # shuffle data & labels
        if shuffle:
            idxs = range(len(X))
            rng.shuffle(idxs)
            X = X[idxs]
            if labels is not None: labels = labels[idxs]
        
        def gen_batch():
            # generate batches
            nb_batch = int(np.ceil(float(X.shape[0])/batch_size))
            for b in range(nb_batch):
                # determine batch size. all should eq batch_size except the last
                # batch of dataset, in cases where len(dataPaths) % batch_size != 0.
                batch_end = (b+1)*batch_size
                if batch_end > X.shape[0]:
                    nb_samples = X.shape[0] - b*batch_size
                else:
                    nb_samples = batch_size
    
                # get a minibatch
                bX = []
                for i in xrange(nb_samples):
                    x = np.array(X[b*batch_size+i], dtype=np.float32)
                    
                    # apply zero-mean and unit-variance, if set
                    x = self._standardize(x)
    
                    # apply augmentations
                    if self.do_static_aug:
                        x = dtf.transform_image(x, output_shape=self.output_shape, 
                                                tf=self.aug_tf,
                                                warp_kwargs=self.static_warp_kwargs)
    
                    if self.do_rng_aug:
                        x = dtf.perturb_image(x, **self.rng_aug_params)
                        
                    bX.append(x)
    
                # clean up minibatch array for return
                bX = np.array(bX, dtype=ret_dtype)
                if ret_chw_order:
                    bX = bX.transpose(0, 3, 1, 2)
                                            
                if labels is not None:
                    yield bX, labels[b*batch_size:b*batch_size+nb_samples]
                else:
                    yield bX
            
        return dtb.buffered_gen_threaded(gen_batch(), buffer_size=buffer_size)
        
    def _set_global_zm(self, X, axis=None):
        """ Sets zero-mean to apply to each minibatch. See `set_zmuv` """
        if self.mean is not None: 
            print("[datagen:BatchGenerator] Warning: Mean was previosuly set. Replacing values...")
        self.do_global_zm = True
        self.mean = X.mean(axis=axis)
        return X - self.mean
        
    def _set_global_uv(self, X, axis=None):
        """ Sets unit-variance to apply to each minibatch. See `set_zmuv` """
        if self.std is not None:
            print("[datagen:BatchGenerator] Warning: Std was previously set. Replacing values...")
        self.do_global_uv = True
        self.std = X.std(axis=axis)
        return X / (self.std + 1e-12)
        
    def _set_samplewise_zm(self, axis=None):
        """ Sets each sample to have zero-mean """
        self.do_samplewise_zm = True
        self.samplewise_zm_axis = axis
    
    def _set_samplewise_uv(self, axis=None):
        """ Sets each sample to have unit-variance """
        self.do_samplewise_uv = True
        self.samplewise_uv_axis = axis
    
    def _standardize(self, x):
        if self.do_global_zm:
            x -= self.mean
        
        if self.do_global_uv:
            x /= (self.std + 1e-12)
        
        if self.do_samplewise_zm:
            x -= np.mean(x, axis=self.samplewise_zm_axis)

        if self.do_samplewise_uv:
            x /= (np.std(x, axis=self.samplewise_uv_axis) + 1e-12)
        
        return x 
    
    def _check_do_params(self):
        if self.do_global_zm:
            if self.mean is None:
                raise Exception("do_global_zm is set but mean is None\n"
                                "... Use `set_global_zmuv` to set mean")
        
        if self.do_global_uv:
            if self.std is None:
                raise Exception("do_global_uv is set but std is None\n"
                                "... Use `set_global_zmuv` to set std")
        
        if self.do_static_aug:
            if self.aug_tf is None:
                raise Exception("do_static_aug is set but aug_tf is None\n"
                                "... use `set_static_aug_params` to set aug params")
            
        if self.do_rng_aug:
            if self.rng_aug_params is None:
                raise Exception("do_rng_aug is set but rng_aug_params is None"
                                "\n... use `set_rng_aug_params` to set rng params")
                                
def default_data_loader(dataPath):
    """ Generic function for loading images. Supports .npy & basic PIL.Image
    compatible extensions. dataPath(str) is the path to the image. """
    # get format of data, using the extension
    import os
    ext = os.path.basename(dataPath).split(os.path.extsep)[1]
    
    # load using numpy
    if ext == '.npy':
        dat = np.load(dataPath)
    # else default to PIL.Image supported extensions. Loads most basic image formats.
    else: 
        try:
            dat = np.array(Image.open(dataPath))
        except IOError:
            raise IOError("default_data_loader does not recognize file type: %s"%ext)
    return dat
    
class DataGenerator(object):
    """
    Iterable batch fetcher with realtime data augmentation.
    Loads the data and applies zero-mean unit variance and augmentations
    on-the-fly. Below, `mb` refers to minibatch.
    
    Parameters
    ---------
    data_loader: function, optional
        Python function to load up the individual images of the dataset. 
        The function should have a call data_loader(dataPath), where 
        dataPath is the path to the image and data_loader returns the 
        image as a ndarray. See `default_data_loader` as an example.
    
    data_loader_kwargs: dict, optional
        Keyword arguments associated with the data_loader function.
        
    do_global_zm: bool, optional
        Subtract mb by mean over the dataset. See `set_global_zmuv`
        
    do_global_uv: bool, optional
        Divide mb by std over the dataset. See `set_global_zmuv`
    
    do_samplewise_zm: bool, optional
        Subtract each sample of the mb by its mean. See `set_samplewise_zmuv`
    
    do_samplewise_uv: bool, optional
        Divide mb each sample of the mb by its std. See `set_samplewise_zmuv`
        
    do_static_aug: bool, optional
        Realtime augment of each mb with stationary augmentations: [crop, zoom, 
        rotation, shear, translation (x,y), flip_lr]. See `set_aug_params`.
        
    do_rng_aug: bool, optional
        Realtime augment of each mb with random augmentation [crop, zoom, 
        rotation, shear, translation (x,y), flip_lr]. See `set_rng_params`.
        
    Call
    ---------
    After initializing the BatchGenerator class, set each do_* procedure with 
    the associated set_* procedures. Then call by creating the generator:
    get_batch( ... ) and calling get_batch.next().
    """
    def __init__(self,
                 data_loader            = default_data_loader,
                 data_loader_kwargs     = {},
                 do_global_zm           = False,
                 do_global_uv           = False,
                 do_samplewise_zm       = False,
                 do_samplewise_uv       = False,
                 do_static_aug          = False,
                 do_rng_aug             = False,
                 ):
            
        self.__dict__.update(locals())
        self.mean                   = None
        self.std                    = None
        self.samplewise_zm_axis     = None
        self.samplewise_uv_axis     = None
        self.aug_tf                 = None
        self.rng_aug_params         = None
        
    def set_data_loader(self, data_loader, data_loader_kwargs={}):
        """ Sets the function used to load images within the minibatches. The 
        function should be of form data_loader(dataPath). See `default_data_loader`"""
        self.data_loader = data_loader
        self.data_loader_kwargs = data_loader_kwargs
        
    def set_global_zmuv(self, mean, std):
        """ Sets global mean and std to apply to every minibatch generation. Use 
        compute_and_set_zmuv to compute and set the zero-mean and zero-std 
        
        Parameters
        ---------
        mean: float
            Mean to substract on each minibatch iteration.
        
        std; float
            Std to divide on each minibatch iteration.
        """
        self._set_global_zm(mean)
        self._set_global_uv(std)
        
    def compute_and_set_global_zmuv(self, dataPaths, batch_size=32, axis=None,
                                    without_augs=True, get_batch_kwargs={}):
        """
        Computes global mean and variance of dataset provided in dataPaths.
        Use set_zmuv if mean and std values are already known.
        
        Parameters
        ---------
        dataPaths: str, semi-optional
            List of paths pointing to the images within the dataset.
            Compute mean and std of minibatches and averages to obtain
            the zero-mean and unit variance mean and std to apply on get_batch.

        batch_size: int, optional
            Size of minibatches to load to compute avg of mean and std.
            
        axis: tuple, int, optional
            Axis to compute over. E.g: axis=0 will compute mean across batch 
            with output shape (height, width, channels). (0,1,2) will compute
            mean across channels with output shape (3,). 
            
        without_augs: bool, optional
            Whether to compute the mean and std via batches with augmentation
            or without. Without_augs=True will load batches without augmentations.
            
        get_batch_kwargs: dict, optional
            Additional keywords to pass to get_batch when loading the minibatches.
            See `DataGenerator.get_batch` for kwargs.
        """
        # compute mean and std without augmentation
        if without_augs:
            do_static_aug = self.do_static_aug
            do_rng_aug = self.do_rng_aug
            self.do_static_aug = False
            self.do_rng_aug = False
            
        # compute and set mean & std
        batches_mean = [] 
        batches_std = []
        
        # set do global zmuv to false so we can iterate thru batches
        self.do_global_uv = False
        self.do_global_zm = False
        for X in self.get_batch(dataPaths, batch_size=batch_size, 
                                shuffle=False, **get_batch_kwargs):
            mean = X.mean(axis=axis)
            X = X - mean
            std = X.std(axis=axis)
            
            batches_mean.append(mean)
            batches_std.append(std)
            
        self.set_global_zmuv(np.mean(batches_mean, axis=0), np.mean(batches_std, axis=0))
        
        # reset augmentation parameters
        if without_augs:
            self.do_static_aug = do_static_aug
            self.do_rng_aug = do_rng_aug
    
    def set_samplewise_zmuv(self, axis=None):
        """
        Sets the axis over which to take the mean and std over each sample
        of the mb to substract and divide. All samples will have zero-mean 
        and unit-variance. See `set_global_umuv` for axis options.
        """
        self._set_samplewise_zm(axis=axis)
        self._set_samplewise_uv(axis=axis)
        
    def set_static_aug_params(self, input_shape, aug_params, warp_kwargs={}):
        """ Sets static augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params is not None and self.do_rng_aug: 
            print("[datagen:DataGenerator] Warning: Static and Random augmentations are both set.")
        
        self.do_static_aug = True
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
        self.output_shape = aug_params.pop('output_shape', None)
        self.static_warp_kwargs = warp_kwargs
        
    def set_rng_aug_params(self, rng_aug_params):
        """ Sets random augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf is not None and self.do_static_aug:
            print("[datagen:DataGenerator] Warning: Static and Random augmentations are both set.")
        
        self.do_rng_aug = True
        self.rng_aug_params = rng_aug_params
        
    def get_batch(self, dataPaths, labels=None, batch_size=32, shuffle=True, buffer_size=2,
                  rng=np.random, ret_opts={'dtype': np.float32, 'chw_order': False}): 
        """ 
        Iterable batch generator. Returns minibatches of the dataset (X, labels) 
        with realtime augmentations, where X is loaded from dataPaths and 
        DataGenerator.data_loader. Use get_batch.next() to fetch batches.

        Augmentation parameters and zmuv need to be set prior to running
        get_batch.        
        
        Parameters
        ---------
        dataPaths: list of str
            Path to images to load in minibatches.
            
        labels: ndarray, shape = (data, ) or (data, one-hot-encoded), optional
            Corresponding labels to dataset. If label is None, get_batch will 
            only return minibatches of X.
        
        batch_size: int, optional
            Sizes of minibatches to extract from X. If X % batch_size != 0,
            then the last batch returned the remainder, X % batch_size.
        
        shuffle: bool, optional
            Whether to shuffle X and labels before generating minibatches
        
        buffer_size: int, optional
            Number of batches to generate in buffer such that subsequent calls
            return a preprocessed batch from the buffer and begin generating more.
            
        rng: np.random.RandomState, optional
            Randomstate to shuffle X (if true) for reproducibility.
        
        ret_opts: dict, optional
            Return options. 'dtype': is the datatype to return per minibatch,
            'chw_order': If true, will return the array in 
            (bsize, channels, height, width), else returns in
            (bsize, height, width, channels).
        """
        # parse ret_opts
        ret_opts = ret_opts.copy() # to not modify original object
        ret_dtype = ret_opts.pop('dtype', np.float32)
        ret_chw_order = ret_opts.pop('chw_order', False)
        
        # check if do_ procedures and params are correct
        self._check_do_params()
        
        # shuffle data & labels
        if shuffle:
            idxs = range(len(dataPaths))
            rng.shuffle(idxs)
            dataPaths = dataPaths[idxs]
            if labels is not None: labels = labels[idxs]

        # generate batches
        def batch_gen():
            ndata = len(dataPaths)        
            nb_batch = int(np.ceil(float(ndata)/batch_size))
            for b in range(nb_batch):
                # determine batch size. all should eq batch_size except the last
                # batch of dataset, in cases where len(dataPaths) % batch_size != 0.
                batch_end = (b+1)*batch_size
                if batch_end > ndata:
                    nb_samples = ndata - b*batch_size
                else:
                    nb_samples = batch_size
                
                # get a minibatch
                bX = []
                for i in xrange(nb_samples):
                    # load data
                    x = np.array(self.data_loader(dataPaths[b*batch_size+i],
                                                  **self.data_loader_kwargs), 
                                                  dtype=np.float32)
    
                    # apply zero-mean and unit-variance, if set
                    x = self._standardize(x)
                        
                    # apply augmentations
                    if self.do_static_aug:
                        x = dtf.transform_image(x, output_shape=self.output_shape, 
                                                tf=self.aug_tf, 
                                                warp_kwargs=self.static_warp_kwargs)
                        
                    if self.do_rng_aug:
                        x = dtf.perturb_image(x, **self.rng_aug_params)
                        
                    bX.append(x)
                
                # clean up minibatch array for return
                bX = np.array(bX, dtype=ret_dtype)
                if ret_chw_order:
                    bX = bX.transpose(0, 3, 1, 2)
                    
                if labels is not None:
                    yield bX, labels[b*batch_size:b*batch_size+nb_samples]
                else:
                    yield bX
            
        return dtb.buffered_gen_threaded(batch_gen(), buffer_size=buffer_size)
        
    def _set_global_zm(self, mean):
        """ Sets zero-mean to apply to each minibatch. See `set_zmuv` """
        if self.mean is not None: 
            print("[datagen:DataGenerator] Warning: Mean was previosuly set. Replacing values...")
        self.do_global_zm = True
        self.mean = mean
        
    def _set_global_uv(self, std):
        """ Sets unit-variance to apply to each minibatch. See `set_zmuv` """
        if self.std is not None:
            print("[datagen:DataGenerator] Warning: Std was previously set. Replacing values...")
        self.do_global_uv = True
        self.std = std

    def _set_samplewise_zm(self, axis=None):
        """ Sets each sample to have zero-mean """
        self.do_samplewise_zm = True
        self.samplewise_zm_axis = axis
    
    def _set_samplewise_uv(self, axis=None):
        """ Sets each sample to have unit-variance """
        self.do_samplewise_uv = True
        self.samplewise_uv_axis = axis

    def _standardize(self, x):
        if self.do_global_zm:
            x -= self.mean
        
        if self.do_global_uv:
            x /= (self.std + 1e-12)
        
        if self.do_samplewise_zm:
            x -= np.mean(x, axis=self.samplewise_zm_axis)

        if self.do_samplewise_uv:
            x /= (np.std(x, axis=self.samplewise_uv_axis) + 1e-12)
        
        return x 
        
    def _check_do_params(self):
        if self.do_global_zm:
            if self.mean is None:
                raise Exception("do_global_zm is set but mean is None\n"
                                "... Use `set_global_zmuv` to set mean")
        
        if self.do_global_uv:
            if self.std is None:
                raise Exception("do_global_uv is set but std is None\n"
                                "... Use `set_global_zmuv` to set std")
        
        if self.do_static_aug:
            if self.aug_tf is None:
                raise Exception("do_static_aug is set but aug_tf is None\n"
                                "... use `set_static_aug_params` to set aug params")
            
        if self.do_rng_aug:
            if self.rng_aug_params is None:
                raise Exception("do_rng_aug is set but rng_aug_params is None"
                                "\n... use `set_rng_aug_params` to set rng params")