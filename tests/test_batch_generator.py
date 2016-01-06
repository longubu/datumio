"""
Test datumio.datagen.BatchGenerator
"""
import datumio.datagen as dtd
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image


class TestGenError(Exception):
    """Raise error if one test fails. Each test usually requires the previous
    to finish without error"""
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return repr(self.msg)


def main(X, y, BATCH_SIZE=32, test_axis=0, show_aug_test=True):
    """Runs a test for each functionality of BatchGenerator

    Parameters
    ----------
    X: iterable of shape (dataset_length, height, width, channels)
        Image dataset to test on.

    y: iterable of shape (dataset_length,)
        Truth for corresponding dataset

    BATCH_SIZE: int, optional
        Size of minibatches per call

    test_axis: None or int, optional
        Axis to test zmuv operations on.
        See numpy.mean for more options for axis

    show_aug_test: bool, optional
        If true, will plot the on-the-fly augmentations (static & random)
        with comparison to the original. This is for manual, visual
        test to see if augmentations are correct (or at least what we expect)

    Returns
    ----------
    None:
        If no exceptions were raised, all tests were ran successfully. However,
        you must check manually check the augmentation tests visually. See
        `show_aug_test`.

    Raises
    ----------
    TestGenError: Exception
        If any functionality fails to do what it's suppose to.
    """

    # ----- test basic io -----#

    # test if minibatches equal to the minibatches we extract manually
    Batcher = dtd.BatchGenerator(X, y=y)
    batchgen = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=False,
                                 chw_order=False, dtype=np.uint8)

    for idx, (mb_x, mb_y) in enumerate(batchgen):
        if not np.all(mb_x == X[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]):
            raise TestGenError("Not all batches in X match correctly from data")
        if not np.all(mb_y == y[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]):
            raise TestGenError("Not all batches in y match correctly from data")

    # test batch shuffling. Assuming the above ^ test runs successfully,
    # we can simply do a != to see if it did any shuffling. Not a sure test.
    Batcher = dtd.BatchGenerator(X, y=y)
    batchgen = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=True,
                                 chw_order=False, dtype=np.uint8)

    for idx, (mb_x, mb_y) in enumerate(batchgen):
        if np.all(mb_x == X[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]):
            raise TestGenError("Batches in X are not correctly shuffled")
        if np.all(mb_y == y[idx*BATCH_SIZE: (idx+1)*BATCH_SIZE]):
            raise TestGenError("Batches in y are not correctly shuffled")

    # test if not providing labels gives us only the batch
    Batcher = dtd.BatchGenerator(X)
    mb_x = batchgen = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=False,
                                        chw_order=False, dtype=np.uint8).next()
    if type(mb_x) == tuple:
        raise TestGenError("Getting batch is returning y when it shouldn't")
    else:
        if mb_x.shape[0] != BATCH_SIZE:
            raise TestGenError("W/out y, but returning incorrect len of batch")

    # get a batch of non-shuffled data
    mb_x_og = X[:BATCH_SIZE]
    mb_y_og = y[:BATCH_SIZE]

    # ----- test zmuv -----#

    # compute dataset-zmuv manually
    tmp_X = np.array(X.copy(), dtype=np.float32)
    mean = tmp_X.mean(axis=test_axis)
    tmp_X = tmp_X - mean
    std = tmp_X.std(axis=test_axis)
    zmuv_mb_x = mb_x_og - mean
    zmuv_mb_x /= std

    # compute dataset-zmuv using BatchGenerator
    Batcher = dtd.BatchGenerator(X, y=y, dataset_mean=True,
                                 dataset_std=True, dataset_axis=test_axis)
    mb_x, mb_y = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=False,
                                   chw_order=False, dtype=np.float32).next()

    if not np.all(mb_x == zmuv_mb_x):
        raise TestGenError("Error correctly generating batch w/ dataset-zmuv")

    # compute batch-zmuv manually
    mb_x_manual = np.array(mb_x_og.copy(), dtype=np.float32)
    mb_x_manual = mb_x_manual - mb_x_manual.mean(axis=test_axis)
    mb_x_manual = mb_x_manual / (mb_x_manual.std(axis=test_axis) + 1e-12)

    # compute batch-zmuv using BatchGenerator
    Batcher = dtd.BatchGenerator(X, y=y, batch_mean=True,
                                 batch_std=True, batch_axis=test_axis)
    mb_x, mb_y = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=False,
                                   chw_order=False, dtype=np.float32).next()

    if not np.all(mb_x == mb_x_manual):
        raise TestGenError("Error correctly generating batch w/ batch-zmuv")

    # compute sample-zmuv manually
    mb_x_manual = []
    for x in mb_x_og:
        ex = np.array(x.copy(), dtype=np.float32)
        ex = ex - ex.mean(axis=test_axis)
        ex = ex / (ex.std(axis=test_axis) + 1e-12)
        mb_x_manual.append(ex)
    mb_x_manual = np.array(mb_x_manual, dtype=np.float32)

    # compute sample-zmuv using BatchGenerator
    Batcher = dtd.BatchGenerator(X, y=y, sample_mean=True,
                                 sample_std=True, sample_axis=test_axis)
    mb_x, mb_y = Batcher.get_batch(batch_size=BATCH_SIZE, shuffle=False,
                                   chw_order=False, dtype=np.float32).next()

    if not np.all(mb_x == mb_x_manual):
        raise TestGenError("Error correctly generating batch with sample zmuv")

    # No test augmentations :-(
    if not show_aug_test:
        return

    # ----- test augmentations -----#

    # set up batch generator with augmentation parameters
    aug_params = dict(
        rotation=15,
        zoom=(1.5, 1.5),
        shear=7,
        translation=(5, -5),
        flip_lr=True,
    )

    Batcher = dtd.BatchGenerator(
                 X, y, input_shape=(32, 32), aug_params=aug_params)

    mb_x_aug, mb_y_aug = Batcher.get_batch(
                            batch_size=BATCH_SIZE, shuffle=False,
                            chw_order=False, dtype=np.uint8).next()

    if not np.all(mb_y_og == mb_y_aug):
        raise TestGenError("Static aug created error in generation of truth")

    # set up batch generator with random augmentations
    rng_aug_params = dict(
        zoom_range=(1/1.5, 1.5),
        rotation_range=(-15, 15),
        shear_range=(-7, 7),
        translation_range=(-5, 5),
        do_flip_lr=True,
        allow_stretch=False,
    )

    Batcher = dtd.BatchGenerator(
                 X, y, input_shape=(32, 32), rng_aug_params=rng_aug_params)

    mb_x_rng, mb_y_rng = Batcher.get_batch(
                            batch_size=BATCH_SIZE, shuffle=False,
                            chw_order=False, dtype=np.uint8).next()

    if not np.all(mb_y_og == mb_y_rng):
        raise TestGenError("Rng augs created error in generation of truth")

    if np.all(mb_x_rng == mb_x_aug) or np.all(mb_x_rng == mb_x_og):
        raise TestGenError("Static or Random augs not correctly done")

    # plot 3 random images from og, static, rng batch sets for comparison
    rng_idxs = np.arange(BATCH_SIZE)
    rng_idxs = np.random.choice(rng_idxs, size=3, replace=False)

    fig, axes = plt.subplots(3, 3, num=1)
    for it, ax in enumerate(axes[:, 0]):
        img = mb_x_og[rng_idxs[it]]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if it == 0: ax.set_title("Original Images")

    for it, ax in enumerate(axes[:, 1]):
        img = mb_x_aug[rng_idxs[it]]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if it == 0: ax.set_title("Static Augmentations")

    for it, ax in enumerate(axes[:, 2]):
        img = mb_x_rng[rng_idxs[it]]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if it == 0: ax.set_title("Random Augmentations")

if __name__ == '__main__':
    # set up data & truth
    data_dir = 'test_data/cifar-10/'
    label_path = 'test_data/cifar-10/labels.csv'
    label_df = pd.read_csv(label_path)
    uids = label_df['uid'].values
    y = label_df['label'].values

    X = [np.array(Image.open(os.path.join(data_dir, uid))) for uid in uids]
    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=int)

    main(X, y, BATCH_SIZE=32, test_axis=0, show_aug_test=False)
    print("[SUCCESS] BatchGenerator passed all tests. Check augs visually")
