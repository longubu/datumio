"""
General useful utility function for data io. Specifically to support `datagen`
"""
import numpy as np
import os
from PIL import Image
from sklearn.utils.class_weight import compute_sample_weight


def img_loader(data_path):
    """ Generic function for loading images. Supports .npy & basic PIL.Image
    compatible extensions.

    Parameters
    ---------
    data_path: str
        Path to the image.

    Returns
    ---------
    img: ndarray
        Loaded image
    """
    # get format of data, using the extension
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
            raise IOError("img_loader does not recognize file ext: %s" % ext)
    return img


def resample_data(arr, weights, sample_fraction=1.0, rng_seed=None):
    """Resamples arr according to weights. This can be used to resample
    unbalanced datasets (either by label, or some external label specified).
    This wraps sklearn.utils.class_weight.compute_sample_weight

    Parameters
    -------
    arr: array-like, shape = [n_samples] or [n_samples, outputs]
        Array of labels/class/group to balance.

    class_weight: dict, list of dicts, "auto", or None, optional
        Weights associated with `arr` in the form ``{label: weight}``, where
        the keys `label` are unique values present in arr and weights are
        the percentage of which to sample. If not given, all classes are
        set to weights of 1. For multi-output problems, a list of dicts
        can be provided in the same order as the columns of y.\n

        The "auto" mode uses the values of y to automatically adjust weights
        inversely proportional to class frequencies in the input data.\n

        For multi-output, the weights of each column of y will be multiplied.

    sample_fraction: float, default=1.0
        Fraction of len(arr) to return when resampling dataset. For example,
        if `sample_fraction=2.0`, will return arr of ``len(2*len(arr))``.

    rng_seed: int, default=None
        Seed to random state that uses np.choice to select idxs of samples.

    Returns
    -------
    idxs: ndarray, shape = (sample_fraction * len(arr))
        Resampled indices of arr according to weights.
    """
    # set randomstate, if supplied
    if rng_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed=rng_seed)

    # check for errors if weights is specified as a dictionary
    if isinstance(weights, dict):
        # check if keys in weights express all unique values present in arr
        uniques = np.unique(arr)
        lbl_weights = np.sort(weights.keys())
        if not np.all(np.equal(lbl_weights, uniques)):
            raise RuntimeError("Unique labels of `arr` are not all contained"
                               "within keys of `weights`. Found \n"
                               "unique(arr) = %s\n"
                               "weights.keys() = %s"
                               % (list(uniques), list(lbl_weights)))

        # now check if the probabilities assigned to weights keys add to 1
        weight_vals = float(np.sum(weights.values()))
        if not weight_vals == 1.0:
            raise RuntimeError("Weight values (probabilities) do not add to "
                               "1.0. Found sum(weight.values()) = %0.5f"
                               % weight_vals)

    # compute sample weights
    sample_weights = compute_sample_weight(weights, arr)
    sample_weights = sample_weights / float(np.sum(sample_weights))

    # resample according to sample_fraction
    idxs = np.arange(len(arr))
    n_data = int(sample_fraction * len(arr))
    ret = rng.choice(idxs, size=n_data, replace=True, p=sample_weights)

    return ret
