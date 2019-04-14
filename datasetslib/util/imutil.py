from __future__ import division

from matplotlib import pyplot as plt
from . import nputil
import numpy as np

LAYOUT_NP = 'np'  # n images of p pixels each
LAYOUT_NHW = 'nhw'
LAYOUT_NHWC = 'nhwc'
LAYOUT_NCHW = 'nchw'


# TODO doesnt work with odd number of subplots

def display_images(images, labels=None, n_cols=8):
    if labels is not None and labels.ndim > 1:
        labels = nputil.argmax(labels)
    if images.ndim == 4 and images.shape[3] == 1:
        images = np.reshape(images, images.shape[0:3])
    n_rows = -(-len(
        images) // n_cols)  # double minus is for upside down floor division to get ceiling division
    for i in range(len(images)):
        plt.subplot(n_rows, n_cols, i + 1)
        if labels is not None:
            plt.title(labels[i])
        plt.imshow(images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def image_layout(x, old, new, **kwargs):
    """
    Note: DO not convert from '4D to 2D to 3D' or '3D to 2D to 4D'
    :param x:
    :param old:
    :param new:
    :param kwargs:
    :return:
    """
    if old == new:
        return x
    else:
        if new == LAYOUT_NP:
            if old == LAYOUT_NHWC:
                h = np.shape[1]
                w = np.shape[2]
                c = np.shape[3]
            elif old == LAYOUT_NCHW:
                h = np.shape[2]
                w = np.shape[3]
                c = np.shape[1]
            elif old == LAYOUT_NHW:
                h = np.shape[1]
                w = np.shape[2]
                c = 1
            p = h * w * c
            x_ = np.reshape(x, [-1, p])
        elif old == LAYOUT_NP:
            h = kwargs.get('h')
            w = kwargs.get('w')
            c = kwargs.get('c')
            if new == LAYOUT_NHWC:
                x_ = np.reshape(x, [-1, h, w, c])
            elif new == LAYOUT_NCHW:
                x_ = np.reshape(x, [-1, c, h, w])
            elif new == LAYOUT_NHW:
                x_ = np.reshape(x, [-1, h, w])
        elif all(layouts in [LAYOUT_NCHW, LAYOUT_NHWC] for layouts in
                 [old, new]):
            new = [old.index(char) for char in new]
            x_ = np.transpose(x, new)
        else:
            raise ValueError("Can't convert from {} to {}".format(old, new))
        if x.shape[0] == x_.shape[0]:
            return x_
        else:
            raise ValueError(
                "Inavalid conversion from {} with shape {} to {} with shape {}".format(
                    old, x.shape, new, x_.shape))
