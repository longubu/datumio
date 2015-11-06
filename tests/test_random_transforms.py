"""
Test random augmentation procedure in datumio.transforms
"""

import datumio.transforms as dtf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# load up image
img_path = 'test_data/cat.jpg'
img = np.array(Image.open(img_path, mode = 'r'))

# set up augmentation parameters
hw = (img.shape[0], img.shape[1])
crop_percentage = 0.5
crop_hw = (hw[0] - int(hw[0]*crop_percentage), hw[1] - int(hw[1]*crop_percentage))

rng = np.random.RandomState(seed=435)
random_augmentation_parmas = [
    {'zoom_range': (1/1.5, 1.5)},
    {'rotation_range': (-30, 30)},
    {'shear_range': (-15, 15)},
    {'translation_range': (-200, 250)},
    {'do_flip_lr': True},
    {'allow_stretch': True, 'zoom_range': (1/1.5, 1.5)},
]
nAugmentations = len(random_augmentation_parmas)

# set up plotting. nPlots = original image + len(augmentations) + all_augment + all_augment_crop
nPlots = nAugmentations + 3
nrows = int(np.ceil(np.sqrt(nPlots)))
ncols = int(np.ceil(nPlots/float(nrows)))

plt.figure(1); plt.clf()
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, num=1)
axes = axes.flatten()

# Plot the original image
axes[0].imshow(img)
axes[0].set_title("Original Image")

# Plot each augmentation parameter, isolated
for it, rng_augmentation_param in enumerate(random_augmentation_parmas):
    ax = axes[it + 1]

    # build transform, then apply fast_warp
    img_wf = dtf.perturb_image(img, rng = rng, **rng_augmentation_param)
    
    # plot image
    ax.imshow(img_wf.astype(np.uint8))
    ax.set_title("%s"%rng_augmentation_param)
    ax.set_xticks([])
    ax.set_yticks([])

# plot image with all augmentations at once
ax = axes[it + 2]
rnd_params = {param.keys()[0]: param.values()[0] for param in random_augmentation_parmas}
img_wf_og = dtf.perturb_image(img, rng=rng, **rnd_params)
ax.imshow(img_wf_og.astype(np.uint8))
ax.set_title("All Augmentations")
ax.set_xticks([])
ax.set_yticks([])

# plot image with all augmentations & center crop
ax = axes[it + 3]
img_wf = dtf.perturb_image(img, output_shape=crop_hw, rng=rng, **rnd_params)
ax.imshow(img_wf.astype(np.uint8))
ax.set_title("All Augmentation + Crop")

# test batch random augmentaitons
imgs = np.array([img, img])
t_imgs = dtf.perturb_images(imgs, ptb_image_kwargs=rnd_params)
plt.figure(); plt.clf()
plt.subplot(311)
plt.title("Batch Rng  TF1")
plt.imshow(t_imgs[0].astype(np.uint8))
plt.subplot(312)
plt.title("Batch Rng TF2")
plt.imshow(t_imgs[1].astype(np.uint8))
plt.subplot(313)
plt.title("Correct All Augmentation")
plt.imshow(img_wf_og.astype(np.uint8))