"""
Collection of data generator classes.

TODO: 
    - multiprocessing
    - is there a way to combine batch generator with dataloader?
        - create a base class 
    - sample wise zero mean , unir var
    - update unit tests
    - make unit tests simpler
        - make unit tests apply imagenet data. larger and more realistic.
        
"""
import numpy as np
from PIL import Image

import transforms as dtf

class BatchGenerator(object):
    """
    Iterable batch fetcher with zmuv & augmentations computed on-the-fly (otf).
    Requires loading the dataset onto memory beforehand.
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    get_batch:
        Batch generator. Computations & augmentations are done.

    set_zmuv: 
        Computes mean and std on all of data. Will apply zmuv
        to each mini batch generation.
    
    set_aug_params: 
        Sets static augmentation parameters
    
    set_rng_aug_params: 
        Sets random augmentation parameter range.
    """
    def __init__(self):
        self.aug_tf         = None
        self.rng_aug_params = None
        self.mean           = None
        self.std            = None

    def get_batch(self, X, labels=None, batch_size=32, shuffle=True, 
                  rng=np.random, ret_opts={'dtype': np.float32, 'chw_order': False}):     
        """ 
        Iterable batch generator, given X data with labels (optional). Augmentations 
        & zmuv are computed on-the-fly. Use get_batch.next() to fetch batches.
        
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
        
        rng: np.random.RandomState, optional
            Randomstate to shuffle X (if true) for reproducibility.
        
        ret_opts: dict, optional
            Return options. 'dtype': is the datatype to return per minibatch,
            'chw_order': If true, will return the array in 
            (bsize, channels, height, width), else returns in
            (bsize, height, width, channels).
        """
        # parse ret_opts
        ret_dtype = ret_opts.pop('dtype', np.float32)
        ret_chw_order = ret_opts.pop('chw_order', False)

        # shuffle data & labels
        if shuffle:
            idxs = range(len(X))
            rng.shuffle(idxs)
            X = X[idxs]
            if labels is not None: labels = labels[idxs]

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
                
                # apply zero-mean and unit-variance
                if (self.mean is not None) and (self.std is not None):
                    x -= self.mean
                    x /= self.std

                # apply augmentations
                if self.aug_tf is not None:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                    
                if self.rng_aug_params is not None:
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
            
    def set_zmuv(self, X, axis=0):
        """
        Computes mean unit std on entire dataset and
        applies to each minibatch generation.
        
        Parameters
        ---------
        X: ndarray, shape = (data, height, width, channels)
            Dataset to generate batch from.
        
        axis: tuple, int, 0 (default)
            Axis to compute over. 0 will compute mean across batch with
            output shape (height, width, channels). (0,1,2) will compute
            mean across channels with output shape (3,). 
        """
        self.mean = X.mean(axis=axis)
        self.std  = X.std(axis=axis)
        
    def set_aug_params(self, input_shape, aug_params):
        """ Sets static augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
            
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
    
    def set_rng_aug_params(self, rng_aug_params):
        """ Sets random augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf
        
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
    Iterable batch fetcher for datasets of large units with zmuv & augmentations 
    computed on-the-fly (otf). Does not require loading the dataset beforehand.
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    set_data_loader:
        Sets the function used to load images within the minibatch.
        
    get_batch:
        Batch generator. Computations & augmentations are done.

    set_zmuv: 
        Computes mean and std on all of data. Will apply zmuv
        to each mini batch generation.
    
    set_aug_params: 
        Sets static augmentation parameters
    
    set_rng_aug_params: 
        Sets random augmentation parameter range.
    """
    
    def __init__(self):
        self.__dict__.update(locals())

        self.aug_tf         = None
        self.rng_aug_params = None
        self.mean           = None
        self.std            = None
        
        self.data_loader = default_data_loader
        self.data_loader_kwargs = {}
        
    def set_data_loader(self, data_loader, data_loader_kwargs={}):
        """ Sets the function used to load images within the minibatches. The 
        function should be of form data_loader(dataPath). See `default_data_loader`"""
        self.data_loader = data_loader
        self.data_loader_kwargs = data_loader_kwargs
       
    def set_aug_params(self, input_shape, aug_params):
        """ Sets static augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
            
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
        
    def set_rng_aug_params(self, rng_aug_params):
        """ Sets random augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params
    
    def set_zmuv(self, mean, std):
        """ Sets zero-mean and zero-std to apply to every minibatch. Use 
        compute_and_set_zmuv to compute and set the zero-mean and zero-std 
        
        Parameters
        ---------
        mean: float
            Mean to substract on each minibatch iteration.
        
        std; float
            Std to divide on each minibatch iteration.
        """
        self.set_zm(mean)
        self.set_uv(std)
        
    def set_zm(self, mean):
        """ Sets zero-mean to apply to each minibatch. See `set_zmuv` """
        if self.mean is not None: 
            raise Warning("Mean was previosuly set. Replacing values...")
        self.mean = mean
        
    def set_uv(self, std):
        """ Sets unit-variance to apply to each minibatch. See `set_zmuv` """
        if self.std is not None:
            raise Warning("Std was previously set. Replacing values...")
        self.std = std
            
    def compute_and_set_zmuv(self, dataPaths, batch_size=32, axis=0,
                             without_augs=True, get_batch_kwargs={}):
        """
        Computes zero-mean and unit variance of dataset provided in dataPaths.
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
            Additional keywords to pass to get_batch when laoding the minibatches.
        """
        # compute mean and std without augmentation
        if without_augs:
            aug_tf = self.aug_tf
            rng_aug_params = self.rng_aug_params
            self.aug_tf = None
            self.rng_aug_params = None
            
        # compute and set mean & std
        batches_mean = [] 
        batches_std = []
        for X in self.get_batch(dataPaths, batch_size=batch_size, 
                                shuffle=False, **get_batch_kwargs):
            batches_mean.append(X.mean(axis=axis))
            batches_std.append(X.std(axis=axis))
            
        self.set_zmuv(np.mean(batches_mean, axis=0), np.mean(batches_std, axis=0))
        
        # reset augmentation parameters
        if without_augs:
            self.aug_tf = aug_tf
            self.rng_aug_params = rng_aug_params
        
    def get_batch(self, dataPaths, labels=None, batch_size=32, shuffle=True, 
                  rng=np.random, ret_opts={'dtype': np.float32, 'chw_order': False}): 
        """ 
        Iterable batch generator, given list of paths to the data and
        associated labels. Loading, Augmentations & zmuv are computed on-the-fly. 
        Use get_batch.next() to fetch batches.
        
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
        
        rng: np.random.RandomState, optional
            Randomstate to shuffle X (if true) for reproducibility.
        
        ret_opts: dict, optional
            Return options. 'dtype': is the datatype to return per minibatch,
            'chw_order': If true, will return the array in 
            (bsize, channels, height, width), else returns in
            (bsize, height, width, channels).
        """
        # parse ret_opts
        ret_dtype = ret_opts.pop('dtype', np.float32)
        ret_chw_order = ret_opts.pop('chw_order', False)
        
        # shuffle data & labels
        if shuffle:
            idxs = range(len(dataPaths))
            rng.shuffle(idxs)
            dataPaths = dataPaths[idxs]
            if labels is not None: labels = labels[idxs]

        # generate batches
        ndata = len(dataPaths)        
        nb_batch = int(np.ceil(float(len(ndata))/self.batch_size))
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
                x = self.data_loader(dataPaths[b*batch_size+i], **self.data_loader_kwargs)
                
                # apply zero-mean and unit-variance
                if (self.mean is not None) and (self.std is not None):
                    x -= self.mean
                    x /= self.std
                
                # apply augmentations
                if self.aug_tf is not None:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                    
                if self.rng_aug_params is not None:
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
