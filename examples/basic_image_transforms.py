"""
Examples showing how to use datumio.transforms to perform basic image
transformations
"""
import datumio.transforms as dtf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_img(img, ax, title=None, imshow_kwargs=None, axis_off=True):
    """Plot image in a convenient way

    Parameters
    ------
    img: ndarray, PIL.Image
        Image to be plotted

    ax: matplotlib.axes
        Canvas to plot on

    title: str, default=None
        String to title plot

    imshow_kwargs: dict
        Keyword arguments to matplotlib.pyplot.imshow

    axis_off: bool, default=True
        Whether to disable ticks on axes.

    Returns
    ------
    None
    """
    if imshow_kwargs is None:
        imshow_kwargs = {}

    if title is None:
        title = ''

    ax.imshow(img, **imshow_kwargs)
    ax.set_title(title, fontsize=20)

    if axis_off:
        ax.set_xticks([])
        ax.set_yticks([])

# load test img
img = np.array(Image.open('../tests/test_data/cat.jpg', 'r'))

# rotate image by 20 deg CCW
img_rot = dtf.transform_image(img, rotation=20)
img_rot = img_rot.astype(np.uint8)  # by default, transform returns float32

# zoom in the center of image by 2x (not upscale). zoom = (col, row)
img_zoom = dtf.transform_image(img, zoom=(2.0, 2.0)).astype(np.uint8)

# translate x,y: -50 in x direction and 200 in y direction
img_txy = dtf.transform_image(img, translation=(-50, 200)).astype(np.uint8)

# Take a 200x200 center-crop of image
img_crop = dtf.transform_image(img, output_shape=(600, 600)).astype(np.uint8)

# rotate, translate, and zoom all at once
img_all = dtf.transform_image(img, rotation=20, zoom=(2.0, 2.0),
                              translation=(-50, 200)).astype(np.uint8)

# view transformations
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
plot_img(img, axes[0, 0], title='Original Image')
plot_img(img_rot, axes[1, 0], title="Image Rotated by 20 deg")
plot_img(img_zoom, axes[2, 0], title="Image zoomed by 2x", axis_off=False)
plot_img(img_txy, axes[0, 1], title="Image translated by (-50, 200)")
plot_img(img_crop, axes[1, 1], title="Image center-cropped to (600, 600)")
plot_img(img_all, axes[2, 1], title="Image with all transformations",
         axis_off=False)
plt.tight_layout()
