"""
Test image transformations from transforms to see if they're doing exactly
what we're want...
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

augmentation_params = dict( rotation = 45,
                            zoom = (0.5, 2.0), # zoom (x, y) = (col, row)
                            shear = 15,
                            translation = (int(hw[0]*0.1), -int(hw[1]*0.2)),
                            flip_lr = True,
                            rescale_shape = (hw[0]/2., hw[1]/2.)
                            )
nAugmentations = len(augmentation_params)

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
for it, (key, param) in enumerate(augmentation_params.iteritems()):
    ax = axes[it + 1]
    augmentation_param = {key: param}
    
    # transform image
    img_wf = dtf.transform_image(img, **augmentation_param)

    # plot image
    ax.imshow(img_wf.astype(np.uint8))
    ax.set_title("Augmentation Param: %s = %s"%(key, param))

# plot image with all augmentations at once
ax = axes[it + 2]
img_wf_og = dtf.transform_image(img, **augmentation_params)
ax.imshow(img_wf_og.astype(np.uint8))
ax.set_title("All Augmentations")
ax.set_xticks([])
ax.set_yticks([])

# plot image with all augmentations & center crop
ax = axes[it + 3]
img_wf = dtf.transform_image(img, output_shape=crop_hw, **augmentation_params)
ax.imshow(img_wf.astype(np.uint8))
ax.set_title("All Augmentation + Crop")

# plot image with some augmentations and rescale to check if rescale works
plt.figure(); plt.clf()
re_aug_params = dict(rotation = 45,
                     translation = (50, -50),
                     flip_lr = True,
                     rescale_shape = (hw[0]/2., hw[1]/2.),
                     )

plt.subplot(311)
img_wf = dtf.transform_image(img, **re_aug_params)
plt.imshow(img_wf.astype(np.uint8))
plt.title("Rescale with some aug, cropped to rescale")
plt.subplot(312)
img_wf = dtf.transform_image(img, output_shape=hw, **re_aug_params)
plt.imshow(img_wf.astype(np.uint8))
plt.title("Rescale with some aug and OUTWARD crop")
plt.subplot(313)
img_wf = dtf.transform_image(img, output_shape=(hw[0]/3., hw[1]/3.), **re_aug_params)
plt.imshow(img_wf.astype(np.uint8))
plt.title("Rescale with some aug and INWARD crop")

# test batch transformations vs regular transformation
imgs = np.array([img, img])
t_imgs = dtf.transform_images(imgs, tf_image_kwargs=augmentation_params)
plt.figure(); plt.clf()
plt.subplot(311)
plt.title("Batch TF1")
plt.imshow(t_imgs[0].astype(np.uint8))
plt.subplot(312)
plt.title("Batch TF2")
plt.imshow(t_imgs[1].astype(np.uint8))
plt.subplot(313)
plt.title("Correct All Augmentation")
plt.imshow(img_wf_og.astype(np.uint8))