"""
Collection of data generator classes.
"""
import numpy as np
import transforms as dtf

class BatchGenerator(object):
    """
    ----
    """
    def __init__(self):
        self.aug_tf         = None
        self.rng_aug_params = None
        self.mean           = None
        self.std            = None
    
    def set_umuv(self, X, axis=None):
        self.set_um(self, X, axis=axis)
        self.set_uv(self, X, axis=axis)
        
    def set_um(self, X, axis=None):
        self.mean = X.mean(axis=axis)
        
    def set_uv(self, X, axis=None):
        self.std = X.std(axis=axis)
        
    def set_aug_params(self, input_shape, aug_params):
        """ input_shape is shape of an image. See datumio.transforms.transform_image for details."""
        if self.rng_aug_params: 
            raise Warning("Warning: Random augmentation is also set. Will do both!")
            
        self.aug_tf = dtf.build_augmentation_transform(input_shape, **aug_params)
    
    def set_rng_aug_params(self, input_shape, rng_aug_params):
        """ input_shape is shape of an image. See datumio.transforms.perturb_image for details."""
        if self.aug_tf:
            raise Warning("Warning: Regular augmentation is also set. Will do both!")
        
        self.rng_aug_params = rng_aug_params # only set parameters instead of build tf
                                                                  
    def get_batch(self, X, y, batch_size=32, shuffle=True, rng=np.random):        
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
                x = X[b*batch_size+i]
                if self.mean:
                    x -= self.mean
                
                if self.std:
                    x /= self.std
                
                if self.aug_tf:
                    x = dtf.transform_image(x, tf=self.aug_tf)
                if self.rng_aug_params:
                    x = dtf.perturb_image(x, **self.rng_aug_params)
                
                bX[i] = x
    
            yield bX, y[b*batch_size:b*batch_size+nb_samples]
        
    