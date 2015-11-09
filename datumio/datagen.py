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
    - take into account greyscale images
    - do more checks to see if self.mean and self.std are both set
    - do axis=None will
            take operation over flattened array.
"""
import numpy as np
from PIL import Image

import transforms as dtf

class BatchGenerator(object):
    """
    Iterable batch fetcher with realtime data augmentation.
    Requires loading the dataset onto memory beforehand.
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    set_zmuv: 
        Computes and sets input mean and std over the dataset (global zero-mean 
        unit-variance). Minibatches wil be substracted by this mean and divided 
        by the std.
    
    set_aug_params: 
        Sets static augmentation parameters. Augmentations include: [crop, zoom,
        rotation, shear, translation (x,y), flip_lr]
    
    set_rng_aug_params: 
        Sets random augmentation parameter range. Random augmentations 
        include: [crop, zoom, rotation, shear, translation (x,y), flip_lr]

    get_batch:
        Batch generator. Fetches batches with realtime augmentation.

    set_zm:
        Computes and set input mean over the dataset.
        
    set_uv: 
        Computes and set input variance over the dataset.
        
    """
    def __init__(self,  # use DataGenerator.set_* to set these parameters
                 do_global_zm=False,
                 do_global_uv=False,
                 do_batchwise_zm=False,
                 do_batchwise_uv=False,
                 do_static_aug=False,
                 do_rng_aug=False,
                 ):
                
        self.__dict__.update(locals())
        self.mean              = None
        self.std               = None
        self.batchwise_zm_axis = None
        self.batchwise_uv_axis = None
        self.aug_tf            = None
        self.rng_aug_params    = None
        
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
        self._set_global_zm(X, axis=axis)
        self._set_global_uv(X, axis=axis)
    
    def set_batchwise_zmuv(self, axis=None):
        self._set_batchwise_zm(axis=axis)
        self._set_batchwise_uv(axis=axis)
    
    def set_static_aug_params(self, input_shape, aug_params):
        """ Sets static augmentation parameters to apply to each minibatch.
        input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params is not None: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
         
        self.do_static_augs = True
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
    
    def set_rng_aug_params(self, rng_aug_params):
        """ Sets random augmentation parameters to apply to each minibatch.
        See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.do_rng_augs = True
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf

    def get_batch(self, X, labels=None, batch_size=32, shuffle=True, 
                  rng=np.random, ret_opts={'dtype': np.float32, 'chw_order': False}):     
        """ 
        Iterable batch generator. Returns minibatches of the dataset (X, labels) with 
        realtime augmentaitons. Use get_batch.next() to fetch batches.
        
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
        
        # check if do_ procedures and params are correct
        self._check_do_params()
        
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
                
                # apply zero-mean and unit-variance, if set
                x = self._standardize(x)

                # apply augmentations
                if self.do_static_augs:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                    
                if self.do_rng_augs:
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
        
    def _set_global_zm(self, X, axis=None):
        """ Sets zero-mean to apply to each minibatch. See `set_zmuv` """
        if self.mean is not None: 
            raise Warning("Mean was previosuly set. Replacing values...")
        self.do_global_zm = True
        self.mean = X.mean(axis=axis)
        
    def _set_global_uv(self, X, axis=None):
        """ Sets unit-variance to apply to each minibatch. See `set_zmuv` """
        if self.std is not None:
            raise Warning("Std was previously set. Replacing values...")
        self.do_global_std = True
        self.std = X.std(axis=axis)
            
    def _set_batchwise_zm(self, axis=None):
        """ """
        self.do_batchwise_zm = True
        self.batchwise_mean_axis = axis
    
    def _set_batchwise_uv(self, axis=None):
        self.do_batchwise_uv = True
        self.batchwise_std_axis = axis
    
    def _standardize(self, x):
        if self.do_global_zm:
            x -= self.mean
        
        if self.do_global_uv:
            x /= self.std
        
        if self.do_batchwise_zm:
            x -= np.mean(x, axis=self.batchwise_mean_axis)
        
        if self.do_batchwise_uv:
            x-= np.std(x, axis=self.batchwise_std_axis)
        
        return x 
    
    def _check_do_params():
        if self.do_global_zm:
            if self.mean is None:
                raise Exception("do_global_zm is set but mean is None\n\
                                ... Use `set_global_zmuv` to set mean")
        
        if self.do_global_uv:
            if self.std is None:
                raise Exception("do_global_uv is set but std is None\n\
                                ... Use `set_global_zmuv` to set std")
        
        if self.do_static_aug:
            if self.aug_tf is None:
                raise Exception("do_static_aug is set but aug_tf is None\n\
                                ... use `set_static_aug_params` to set aug params")
            
        if self.do_rng_aug:
            if self.rng_aug_parmas is None:
                raise Exception("do_rng_aug is set but rng_aug_params is None\n\
                                ... use `set_rng_aug_params` to set rng params")
        
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
    Iterable batch fetcher with realtime data augmentation. Loads the data, sets
    zero-mean unit variance and augmentations on-the-fly.
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    set_data_loader:
        Sets the function used to load images.

    set_zmuv: 
        Sets input global mean and std (zero-mean unit variance).
        Minibatches wil be substracted by this mean and divided 
        by the std.
    
    compute_and_set_zmuv:
        Computes and sets input mean and std over the dataset (global zero-mean 
        unit-variance). 
        
    set_aug_params: 
        Sets static augmentation parameters. Augmentations include: [crop, zoom,
        rotation, shear, translation (x,y), flip_lr]
    
    set_rng_aug_params: 
        Sets random augmentation parameter range. Random augmentations 
        include: [crop, zoom, rotation, shear, translation (x,y), flip_lr]

    get_batch:
        Batch generator. Fetches batches with realtime augmentation.
        
    set_zm:
        Sets input mean over the dataset.
    
    set_uv:
        Sets unit variance over the dataset.
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
       
    def set_zmuv(self, mean, std):
        """ Sets global mean and std to apply to every minibatch generation. Use 
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
        
    def compute_and_set_zmuv(self, dataPaths, batch_size=32, axis=0,
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
        
    def get_batch(self, dataPaths, labels=None, batch_size=32, shuffle=True, 
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
                if self.mean is not None:
                    x -= self.mean
                
                if self.std is not None:
                    x /= self.std
                
                if self.do_batchwise_zm:
                    
                if self.do_batchsise_uv:
                    
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