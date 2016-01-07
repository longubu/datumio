"""
Test random augmentation procedure in datumio.transforms
"""
import datumio.transforms as dtf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(img):
    """Applies random transformations to img and plots for visual comparison"""
    # set up augmentation parameters
    hw = (img.shape[0], img.shape[1])
    crop_percentage = 0.1
    crop_hw = (hw[0] - int(hw[0]*crop_percentage),
               hw[1] - int(hw[1]*crop_percentage))

    rng = np.random.RandomState(seed=17)
    rng_aug_params = [
        {'zoom_range': (1/1.5, 1.5)},
        {'rotation_range': (-30, 30)},
        {'shear_range': (-15, 15)},
        {'translation_range': (-200, 250)},
        {'do_flip_lr': True},
        {'allow_stretch': True, 'zoom_range': (1/1.5, 1.5)}
    ]
    nAugmentations = len(rng_aug_params)

    # set up plotting.
    # nPlots = original image + len(augs) + all_augment + all_augment_crop
    nPlots = nAugmentations + 3
    nrows = int(np.ceil(np.sqrt(nPlots)))
    ncols = int(np.ceil(nPlots/float(nrows)))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 12))
    axes = axes.flatten()

    # Plot the original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    # Plot each augmentation parameter, isolated
    for it, rng_augmentation_param in enumerate(rng_aug_params):
        ax = axes[it + 1]

        # build transform, then apply fast_warp
        img_wf = dtf.perturb_image(img, rng=rng, **rng_augmentation_param)

        # plot image
        ax.imshow(img_wf.astype(np.uint8))
        ax.set_title("%s" % rng_augmentation_param)
        ax.set_xticks([])
        ax.set_yticks([])

    # plot image with all augmentations at once
    rnd_params = {}
    for param in rng_aug_params:
        rnd_params[param.keys()[0]] = param.values()[0]

    ax = axes[it + 2]
    img_wf_og = dtf.perturb_image(img, rng=rng, **rnd_params)
    ax.imshow(img_wf_og.astype(np.uint8))
    ax.set_title("All Augmentations")
    ax.set_xticks([])
    ax.set_yticks([])

    # plot image with all augmentations & center crop
    ax = axes[it + 3]
    img_wf = dtf.perturb_image(img, output_shape=crop_hw,
                               rng=rng, **rnd_params)
    ax.imshow(img_wf.astype(np.uint8))
    ax.set_title("All Augmentation + Crop")

if __name__ == '__main__':
    # load test image
    img_path = 'test_data/cat.jpg'
    img = np.array(Image.open(img_path, mode='r'))

    main(img)
    print("Finished running tests. Check figure to check transformations")
