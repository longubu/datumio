"""
datumio.transforms
---------
General image processing functions that wrap scikit-image core functions
more cleanly and intuitively. Includes functions for performing
static and random transformations on images.

Possible Transforms
---------
- crop
- zoom x, zoom y
- rotation
- translation x, translation y
- flip left/right
- flip up/down
- above transforms with image wrapping/constant padding, etc.

General use functions
---------
- transform_image: Transform an image given set augmentation arguments
- perturb_image: Randomly transforms an image given ranges specified in
    `augmentation` args

Many of the functions were copied and modified from Kaggle National
Data Science Bowl's 1st place winners.
Original in: https://github.com/benanne/kaggle-ndsb/blob/master/buffering.py

COPYRIGHT
---------
All contributions by Sander Dielman:
Copyright (c) 2015 Sander Dieleman
All rights reserved.

All contributions by Long Van Ho:
Copyright (c) 2015 Long Van Ho
All rights reserved.

All other contributions:
Copyright (c) 2015, the respective contributors.
All rights reserved.

LICENSE
---------
The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN
"""
import numpy as np
import skimage.transform


def transform_image(img, output_shape=None, tf=None, zoom=(1.0, 1.0),
                    rotation=0., shear=0., translation=(0, 0), flip_lr=False,
                    flip_ud=False, warp_kwargs=None):
    """Transforms image by cropping (output_shape), zoom, rotation, shear,
    translations, and flips (ud/lr). Crops image if output_shape is not None.

    Parameters
    ---------
    img: ndarray
        Input image to be transformed of shape (row, col, channels)

    tf: skimage.transform._geometric.SimilarityTransform, optional
        A built transformation (matrix), containing all the affine transforms
        to be applied to `img`. If set, will override all other aug params.

    output_shape: iterable of length 2 of dtype int, optional
        Center-crop shape of the resulting output transformed image. For
        transformed images, rotations/zooms typically create regions of
        unnecessary pixels -- this remove the boundaries of the iamge.
        If None (default), output_shape = input_shape

    zoom: iterable of length 2 of dtype float, optional
        E.g: (zoom_row, zoom_col). Scale image rows by zoom_row and image cols
        by zoom_col. Float of < 1 indicate zoom out, >1 indicate zoom in.

    rotation: float, optional
        Rotation angle in counter-clockwise direction as `degrees`.
        Note: Original skimage implementation requires rotation in `rad`.

    shear: float, optional
        Shear angle in counter-clockwise direction as `degrees`.
        Note: Original skimage implementation requires shear in `rad`.

    translation: iterable of length 2 of dtype ints, optional
        Translates image in (x, y). Positive values translates up/right.

    flip_lr: bool, optional
        Flip image left/right

    flip_ud: bool, optional
        Flip image up/down

    warp_kwargs: dict, optional
        Keyword arguments to be sent to fast_warp.
        See `datumio.transforms.fast_warp`

    Returns
    ---------
    img_wf: ndarray of dtype = np.float32
        Transformed image with shape = `output_shape`. If output_shape is None,
        will return transformed image of same dimension as input.
    """
    if tf is None:
        input_shape = img.shape[:2]
        tf = build_augmentation_transform(
            input_shape, output_shape=output_shape, zoom=zoom,
            rotation=rotation, shear=shear, translation=translation,
            flip_lr=flip_lr, flip_ud=flip_ud)

    if warp_kwargs is None:
        warp_kwargs = {}

    return fast_warp(img, tf, output_shape=output_shape, **warp_kwargs)


def perturb_image(img, output_shape=None, zoom_range=(1.0, 1.0),
                  rotation_range=(0., 0.), shear_range=(0., 0.),
                  translation_range=(0, 0), do_flip_lr=False, do_flip_ud=False,
                  allow_stretch=False, rng=None, warp_kwargs=None):
    """Randomly transforms image by cropping (output_shape), zoom, rotation,
    shear, translations, and flips (ud/lr). Crops image if output_shape
    is not None.

    Parameters
    ---------
    img: ndarray
        Input image to be transformed of shape (row, col, channels)

    output_shape: iterable of length 2 of dtype int, optional
        Center-crop shape of the resulting output transformed image. For
        transformed images, rotations/zooms typically create regions of
        unnecessary pixels -- this remove the boundaries of the iamge.
        If None (default), output_shape = input_shape

    zoom_range: iterable of length 2 of dtype float, optional
        E.g: (zoom_low, zoom_high). Will zoom randomly in x & y
        If allow_stretch = True, then x and y will be zoomed individually

    rotation_range: iterable of length 2 of dtype float, optional
        E.g: (low_deg, high_deg). Will rotate CCW by an angle chosen between
        randomly in the range supplied. Angles are in `deg` between (-180, 180]

    shear_range: iterable of length 2 of dtype float, optional
        Randomly applies shear ccw between `shear_range`. Shear angles are in
        `deg` between (-180, 180]

    translation_range: iterable of length 2 of dtype int, optional
        E.g: (low_pixel, high_pixel). Randomly trasnslates x, y pixels
        between the range specified.

    do_flip_lr: bool, optional
        Randomly flip the image symetrically left and right

    do_flip_ud: bool, optional
        Randomly flip the image symetrically up and down

    allow_stretch: bool, float, optional
        Allows zoom x & y to be indepdently chosen.
        If bool, will stretch x & y randomly between `log(zoom_range)`.
        If float, will randomly choose zoom, then also choose a random stretch
        in [-np.log(stretch), +np.log(stretch)]. Will then inversely stretch
        x & y by zoom*stretch and zoom/stretch respectively.

    rng: mtrand.RandomState, optional
        Random state given by np.random.Randomstate() for reproducibility.

    warp_kwargs: dict, optional
        Keyword arguments to be sent to fast_warp.
        See datumio.transforms.fast_warp

    Returns
    ---------
    img_wf: ndarray of dtype = np.float32
        Transformed image with shape = `output_shape`. If output_shape is None,
        will return transformed image of same dimension as input.
    """
    if rng is None:
        rng = np.random

    input_shape = img.shape[:2]
    tf = build_random_augmentation_transform(
        input_shape, output_shape=output_shape, zoom_range=zoom_range,
        rotation_range=rotation_range, shear_range=shear_range,
        translation_range=translation_range, do_flip_lr=do_flip_lr,
        do_flip_ud=do_flip_ud, allow_stretch=allow_stretch, rng=rng)

    if warp_kwargs is None:
        warp_kwargs = {}

    return transform_image(img, output_shape=output_shape, tf=tf,
                           warp_kwargs=warp_kwargs)


def fast_warp(img, tf, output_shape=None, order=1, mode='constant', cval=0):
    """This wrapper function is faster than skimage.transform.warp. Transforms
    an image based on built skimage similiarity transform matrix.

    Parameters
    ---------
    img: ndarray
        Input image to be transformed

    tf: skimage.transform._geometric.SimilarityTransform
        A built skimage.transform.SimilarityTransform, containing all the
        affine transformations to be applied to `img`.

    output_shape: iterable of length 2 of dtype int, optional
        Center-crop :math:`tf(img)` to dimensions of `output_shape`.
        If None (default), output_shape = (`img.shape[0]`, `img.shape[1]`).

    order: int, optional
        The order of interpolation. The order has to be in the range 0-5.

         - 0: Nearest-neighbor
         - 1: Bi-linear (default)
         - 2: Bi-quadratic
         - 3: Bi-cubic
         - 4: Bi-quartic
         - 5: Bi-quintic

    mode: str, optional
        Points outside the boundaries of the input are filled according to the
        given mode('constant', 'nearest', 'reflect', 'wrap').

    cval: int, optional
        Value to fill for points outside boundaries.

    Returns
    ---------
    warped: ndarray of np.float32
        Transformed img with size given by `output_shape`. If
        `output_shape=None`, returns warped img of input image shape.
    """
    m = tf.params  # tf._matrix is deprecated. m is a 3x3 matrix
    if len(img.shape) < 3:  # if image is greyscale
        img_wf = skimage.transform._warps_cy._warp_fast(
            img, m, output_shape=output_shape, order=order,
            mode=mode, cval=cval)

    else:  # if image is not greyscale, e.g. RGB, RGBA, etc.
        nChannels = img.shape[-1]

        if output_shape is None:
            output_shape = (img.shape[0], img.shape[1])

        img_wf = np.empty((output_shape[0], output_shape[1], nChannels),
                          dtype='float32')  # (height, width, channels)

        for k in xrange(nChannels):
            img_wf[..., k] = skimage.transform._warps_cy._warp_fast(
                img[..., k], m, output_shape=output_shape,
                mode=mode, cval=cval)

    return img_wf


# build affine transforms


def build_centering_transform(image_shape, output_shape):
    """Builds a transform that shifts the center of the `image_shape` to
    center of `output_shape`.

    Parameters
    ---------
    image_shape: iterable of length 2 of dtype int
        Input image shape to be re-centered

    output_shape: interable of length 2 of dtype int
        Input image is recentered to the center of `output_shape`

    Returns
    ---------
    tf: skimage.transform._geometric.SimilarityTransform
        Built affine transformation.
    """
    rows, cols = image_shape
    trows, tcols = output_shape
    shift_x = (cols - tcols) / 2.0
    shift_y = (rows - trows) / 2.0
    return skimage.transform.SimilarityTransform(
        translation=(shift_x, shift_y))


def build_center_uncenter_transforms(image_shape):
    """These are used to ensure that zooming and rotation happens around
    the center of the image. Use this transform to center and uncenter the
    image around such a transform.

    Parameters
    ---------
    image_shape: iterable of length 2 of type int
        Input image shape of (row, col)

    Returns
    ---------
    tf: skimage.transform._geometric.SimilarityTransform
        Built affine transformation.
    """
    # need to swap rows and cols here apparently! confusing!
    center_shift = np.array([image_shape[1], image_shape[0]]) / 2.0 - 0.5

    tform_uncenter = skimage.transform.SimilarityTransform(
        translation=-center_shift)

    tform_center = skimage.transform.SimilarityTransform(
        translation=center_shift)

    return tform_center, tform_uncenter


def build_augmentation_transform(input_shape, output_shape=None,
                                 zoom=(1.0, 1.0), rotation=0., shear=0.,
                                 translation=(0, 0), flip_lr=False,
                                 flip_ud=False):
    """Wrapper to build an affine transformation matrix applies:
    [zoom, rotate, shear, translate, and flip_lr, flip_ud]

    The original skimage implementation applies the transformations to
    bottom left of the image, instead of the center. This wrapper
    centers/uncenters accordingly to apply all transformations correctly WRT
    center of the image.

    See skimage.transform.AffineTransform for more details.

    Parameters
    ---------
    input_shape: iterable of length 2 of dtype int
        Input image shape of form (rows, cols) to be transformed.

    output_shape: iterable of length 2 of dtype int, optional
        Center-crop shape of the resulting output transformed image. For
        transformed images, rotations/zooms typically create regions of
        unnecessary pixels -- this remove the boundaries of the iamge.
        If None (default), output_shape = input_shape

    zoom: iterable of length 2 of dtype float, optional
        E.g: (zoom_row, zoom_col). Scale image rows by zoom_row and image cols
        by zoom_col. Float of < 1 indicate zoom out, >1 indicate zoom in.

    rotation: float, optional
        Rotation angle in counter-clockwise direction as `degrees`.
        Note: Original skimage implementation requires rotation in `rad`.

    shear: float, optional
        Shear angle in counter-clockwise direction as `degrees`.
        Note: Original skimage implementation requires shear in `rad`.

    translation: iterable of length 2 of dtype ints, optional
        Translates image in (x, y). Positive values translates up/right.

    flip_lr: bool, optional
        Flip image left/right

    flip_ud: bool, optional
        Flip image up/down

    Returns
    ---------
    tf: skimage.transform._geometric.SimilarityTransform
        Built affine transformation.

    """
    if flip_lr:
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    if flip_ud:
        shear += 180
        rotation += 360
        # this combination will flip the image upside down

    # A negative x SHOULD move the image to the left by x.
    # Skimage default does otherwise.
    xt = -1*translation[0]

    if output_shape is None:
        output_shape = input_shape

    tf_centering = build_centering_transform(input_shape, output_shape)
    tf_center, tf_uncenter = build_center_uncenter_transforms(input_shape)
    tf_augment = skimage.transform.AffineTransform(
        scale=(1/float(zoom[0]), 1/float(zoom[1])),
        rotation=np.deg2rad(rotation), shear=np.deg2rad(shear),
        translation=(xt, translation[1]))

    # order of addition matters
    tf = tf_centering + tf_uncenter + tf_augment + tf_center
    return tf


def build_random_augmentation_transform(
        input_shape, output_shape=None, zoom_range=(1.0, 1.0),
        rotation_range=(0., 0.), shear_range=(0, 0), translation_range=(0, 0),
        do_flip_lr=False, do_flip_ud=False, allow_stretch=False, rng=None):
    """Randomly perturbs image using affine transformations.

    Parameters
    ---------
    input_shape: iterable of length 2 of dtype of int
        Input image shape of form (rows, cols) to be transformed.

    output_shape: iterable of length 2 of dtype int, optional
        Center-crop shape of the resulting output transformed image. For
        transformed images, rotations/zooms typically create regions of
        unnecessary pixels -- this remove the boundaries of the iamge.
        If None (default), output_shape = input_shape

    zoom_range: iterable of length 2 of dtype float, optional
        E.g: (zoom_low, zoom_high). Will zoom randomly in x & y
        If allow_stretch = True, then x and y will be zoomed individually

    rotation_range: iterable of length 2 of dtype float, optional
        E.g: (low_deg, high_deg). Will rotate CCW by an angle chosen between
        randomly in the range supplied. Angles are in `deg` between (-180, 180]

    shear_range: iterable of length 2 of dtype float, optional
        Randomly applies shear ccw between `shear_range`. Shear angles are in
        `deg` between (-180, 180]

    translation_range: iterable of length 2 of dtype int, optional
        E.g: (low_pixel, high_pixel). Randomly trasnslates x, y pixels
        between the range specified.

    do_flip_lr: bool, optional
        Randomly flip the image symetrically left and right

    do_flip_ud: bool, optional
        Randomly flip the image symetrically up and down

    allow_stretch: bool, float, optional
        Allows zoom x & y to be indepdently chosen.
        If bool, will stretch x & y randomly between `log(zoom_range)`.
        If float, will randomly choose zoom, then also choose a random stretch
        in [-np.log(stretch), +np.log(stretch)]. Will then inversely stretch
        x & y by zoom*stretch and zoom/stretch respectively.

    rng: mtrand.RandomState, optional
        Random state given by np.random.Randomstate() for reproducibility.

    Returns
    ---------
    tf: skimage.transform._geometric.SimilarityTransform
        Built affine transformation.
    """
    if rng is None:
        rng = np.random

    shift_x = int(rng.uniform(*translation_range))
    shift_y = int(rng.uniform(*translation_range))
    translation = (shift_x, shift_y)

    rotation = rng.uniform(*rotation_range)
    shear = rng.uniform(*shear_range)

    if do_flip_lr:
        flip_lr = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip_lr = False

    if do_flip_ud:
        flip_ud = (rng.randint(2) > 0)  # flip half of the time
    else:
        flip_ud = False

    # random zoom
    log_zoom_range = [np.log(z) for z in zoom_range]
    if isinstance(allow_stretch, float):
        log_stretch_range = [-np.log(allow_stretch), np.log(allow_stretch)]
        zoom = np.exp(rng.uniform(*log_zoom_range))
        stretch = np.exp(rng.uniform(*log_stretch_range))
        zoom_x = zoom * stretch
        zoom_y = zoom / stretch
    elif allow_stretch is True:  # avoid bugs, f.e. when it is an integer
        zoom_x = np.exp(rng.uniform(*log_zoom_range))
        zoom_y = np.exp(rng.uniform(*log_zoom_range))
    else:
        # the range should be multiplicatively symmetric, so [1/1.1, 1.1]
        # instead of [0.9, 1.1] makes more sense.
        zoom_x = zoom_y = np.exp(rng.uniform(*log_zoom_range))

    return build_augmentation_transform(
        input_shape=input_shape, output_shape=output_shape,
        zoom=(zoom_x, zoom_y), rotation=rotation, shear=shear,
        translation=translation, flip_lr=flip_lr, flip_ud=flip_ud)
