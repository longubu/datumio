"""
Collection of data generator classes.
"""
import numpy as np
import transforms as dtf

class BatchGenerator(object):
    """
    Iterable batch fetcher with umuv & augmentations computed on-the-fly (otf).
    
    Attributes: See documentation for each to see parameters & use.
    ---------
    get_batch:
        Batch generator. Computations & augmentations are done.

    set_umuv: 
        Computes & set unit mean unit std. Batches will be umuv-ed.
    
    set_aug_params: 
        Sets static augmentation parameters. Batches will be 
        augmented.
    
    set_rng_aug_params: 
        Sets random augmentation parameter range. Batches will be
        perturbed. 

    set_um: 
        Computes & set unit mean of dataset. Batches will be um-ed.
    
    set_uv: 
        Computes & set unit variance of dataset. Batches will be uv-ed.    
        
    """
    def __init__(self):
        self.aug_tf         = None
        self.rng_aug_params = None
        self.mean           = None
        self.std            = None

    def get_batch(self, X, y=None, batch_size=32, shuffle=True, rng=np.random):     
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
                
        """
        if shuffle:
            idxs = range(len(X))
            rng.shuffle(idxs)
            X = X[idxs]
            y = y[idxs]

        nb_batch = int(np.ceil(float(X.shape[0])/batch_size))
        for b in range(nb_batch):
            batch_end = (b+1)*batch_size
            if batch_end > X.shape[0]:
                nb_samples = X.shape[0] - b*batch_size
            else:
                nb_samples = batch_size
            
            bX = np.zeros( [nb_samples] + list(X.shape)[1:])
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
                
                bX[i] = x
    
            yield bX, y[b*batch_size:b*batch_size+nb_samples]
            
    def set_umuv(self, X, axis=0):
        """
        Computes & set unit mean unit std. Batches will be umuv-ed otf.
        
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
    
    def set_rng_aug_params(self, input_shape, rng_aug_params):
        """ input_shape is shape of an image. See datumio.transforms.perturb_image."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf
        
    def set_um(self, X, axis=0):
        """ Computes & set unit mean. Batches will be um-ed otf. See `set_umuv` """
        self.mean = X.mean(axis=axis)
        
    def set_uv(self, X, axis=0):
        """ Computes & set unit std. Batched with be uv-ed otf. See `set_umuv` """
        self.std = X.std(axis=axis)
    