"""
Collection of data generator classes.

TODO: 
    - on-the-fly DataLoader
    - multiprocessing
    - convert batchGenerator to do vector-like applications
    - Tests if faster to load batch then tf, or load images 1by1 tf
    
"""
import numpy as np
from PIL import Image

import transforms as dtf

class BatchGenerator(object):
    """
    Iterable batch fetcher with umuv & augmentations computed on-the-fly (otf).
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    get_batch:
        Batch generator. Computations & augmentations are done.

    set_umuv: 
        Computes & set unit mean unit std on all of data & 
        applies to each mini batch generation.
    
    set_aug_params: 
        Sets static augmentation parameters
    
    set_rng_aug_params: 
        Sets random augmentation parameter range.

    set_um: 
        Computes & set unit mean of dataset. 
        
    set_uv: 
        Computes & set unit variance of dataset. 
        
    """
    def __init__(self):
        self.aug_tf         = None
        self.rng_aug_params = None
        self.mean           = None
        self.std            = None

    def get_batch(self, X, y=None, batch_size=32, shuffle=True, rng=np.random, return_chw_order=False):     
        """ 
        Iterable batch generator, given X data with y labels. Augmentations & umuv are 
        computed on the fly. Use get_batch.next() to fetch batches.
        
        Parameters
        ---------
        X: ndarray, shape = (data, height, width, channels)
            Dataset to generate batch from.
            
        y: ndarray, shape = (data, ) or (data, one-hot-encoded-labels), optional
            Corresponding labels to dataset. If y is None, get_batch will 
            only return batches of X (for datasets with no labels).
        
        batch_size: int, optional
            Sizes of minibatches to extract from X. If X % batch_size != 0,
            then the last batch returned by get_batch is X % batch_size.
        
        shuffle: bool, optional
            Whether to shuffle X before generating minibatches
        
        rng: np.random.RandomState
            Randomstate to shuffle X (if true) for reproducibility.
        
        return_chw_order: bool, optional
            If True, returns minibatch in dimensions (channels, height, width).
        """
        if shuffle:
            idxs = range(len(X))
            rng.shuffle(idxs)
            X = X[idxs]
            if y is not None: y = y[idxs]

        nb_batch = int(np.ceil(float(X.shape[0])/batch_size))
        for b in range(nb_batch):
            batch_end = (b+1)*batch_size
            if batch_end > X.shape[0]:
                nb_samples = X.shape[0] - b*batch_size
            else:
                nb_samples = batch_size
            
            bX = []
            for i in xrange(nb_samples):
                x = np.array(X[b*batch_size+i], dtype=np.float32)
                if self.mean is not None:
                    x -= self.mean
                
                if self.std is not None:
                    x /= self.std
                
                if self.aug_tf is not None:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                    
                if self.rng_aug_params is not None:
                    x = dtf.perturb_image(x, **self.rng_aug_params)
                
                bX.append(x)
            
            bX = np.array(bX)
            if return_chw_order:
                bX = bX.transpose(0, 3, 1, 2)
                
            if y is not None:
                yield bX, y[b*batch_size:b*batch_size+nb_samples]
            else:
                yield bX
            
    def set_umuv(self, X, axis=0):
        """
        Computes & set unit mean unit std on entire dataset and
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
        self.set_um(X, axis=axis)
        self.set_uv(X, axis=axis)

    def set_aug_params(self, input_shape, aug_params):
        """ input_shape is shape of an image. See datumio.transforms.transform_image."""
        if self.rng_aug_params: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
            
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
    
    def set_rng_aug_params(self, rng_aug_params):
        """ input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf
        
    def set_um(self, X, axis=0):
        """ Computes & set unit mean. See `set_umuv` """
        self.mean = X.mean(axis=axis)
        
    def set_uv(self, X, axis=0):
        """ Computes & set unit std. See `set_umuv` """
        self.std = X.std(axis=axis)

def default_data_loader(dataPath):
    return np.array(Image.open(dataPath), dtype=np.float32)
    
def DataGenerator(object):
    """
    ---
    """
    def __init__(self, data_loader=None, data_loader_kwargs={}):
        self.__dict__.update(locals())

        self.aug_params = None
        self.rng_aug_params = None
        self.mean = None
        self.std = None
        
        if data_loader is None:
            self.data_loader = default_data_loader
        else:
            self.data_loader = data_loader
            
    def set_aug_params(self, input_shape, aug_params):
        """ input_shape is shape of an image. See datumio.transforms.transform_image."""
        #TODO: FIX THIS AND RNG ONE
        if self.rng_aug_params: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
            
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
        
    def set_rng_aug_params(self, rng_aug_params):
        """ input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params
    
    def set_umuv(self, mean=None, std=None, dataPaths=None, axis=0, batch_size=32):
        """ """
        compute_mean = True
        compute_std = True
        
        if mean is not None:
            self.mean = float(mean)
            compute_mean = False
        
        if std is not None:
            self.std = float(std)
            compute_std = False
        
        if (compute_mean or compute_std) and dataPaths is not None:
            if compute_mean: mean = []
            if compute_std: std = []
               
            for X in get_batch(dataPaths, batch_size=batch_size, shuffle=False):
                if compute_std:
                    std.append(X.std(axis=axis))
                if compute_mean:
                    mean.append(X.mean(axis=axis))
            if compute_std: self.std = np.mean(std)
            if compute_mean: self.mean = np.mean(mean)
           #TODO: write umuv fnc
        else:
           raise Warning("Need to supply dataPaths or (std & mean) in order \
                           to set the unit-mean unit-variance")
                               
    def set_um(self, mean=None, dataPaths=None, batch_size=32):
        """ """
        if mean is not None:
            self.mean = float(mean)
        elif dataPaths is not None:
            pass
            #TODO: write um fnc
        else:
            raise Warning("Need to supply one of two: mean or dataPaths in order \
                            to set the unit-mean")
    
    def set_uv(self, std=None, dataPaths=None, batch_size=32):
        if std is not None:
            self.std = float(std)
        elif dataPaths is not None:
            pass
            #TODO: write uv fnc
        else:
            raise Warning("Need to supply one of two: mean or dataPaths in order \
                            to set the unit-variance")
                            
    
    def get_batch(self, dataPaths, labels=None, batch_size=32, shuffle=True, 
                  rng=np.random, return_chw_order=False):
        
        if shuffle:
            idxs = range(len(dataPaths))
            rng.shuffle(idxs)
            dataPaths = dataPaths[idxs]
            if labels is not None: labels = labels[idxs]

        ndata = len(dataPaths)        
        nb_batch = int(np.ceil(float(len(ndata))/self.batch_size))
        for b in range(nb_batch):
            batch_end = (b+1)*batch_size
            if batch_end > ndata:
                nb_samples = ndata - b*batch_size
            else:
                nb_samples = batch_size
            
            bX = []
            for i in xrange(nb_samples):
                x = self.data_loader(dataPaths[b*batch_size+i], **self.data_loader_kwargs)
                if self.mean is not None:
                    x -= self.mean
                
                if self.std is not None:
                    x /= self.std
                
                if self.aug_tf is not None:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                    
                if self.rng_aug_params is not None:
                    x = dtf.perturb_image(x, **self.rng_aug_params)
                
                bX.append(x)
            
            bX = np.array(bX)
            if return_chw_order:
                bX = bX.transpose(0, 3, 1, 2)
                
            if labels is not None:
                yield bX, labels[b*batch_size:b*batch_size+nb_samples]
            else:
                yield bX
            