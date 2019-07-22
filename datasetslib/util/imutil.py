from __future__ import division

from matplotlib import pyplot as plt
from . import nputil
import numpy as np

try:  # py3
    from urllib.parse import urlparse
    from urllib import request
except:  # py2
    from urlparse import urlparse
    from six.moves.urllib import request

import tarfile
import itertools
import cv2

LAYOUT_NP = 'np'  # n images of p pixels each
LAYOUT_NHW = 'nhw'
LAYOUT_NHWC = 'nhwc'
LAYOUT_NCHW = 'nchw'


# TODO doesnt work with odd number of subplots
# Always supply 4 dim array : n, h, w, c
# if sending single image use np.expand_dims(img,axis=0)
# convert one hot labels with nputil.argmax(labels)
def image_display(images, labels=[], n_cols=5, figsize=(8,8)):
    if images.ndim < 4:
        raise Exception('image array is not 4D')
    if images.shape[3] > 3:
        raise Exception('3+ channels not supported')
    if images.shape[3] == 1:
        images = np.squeeze(images,axis=3) # remove the last axis if it is 1
        #images.reshape(images.shape[0:3])
    n_rows = -(-images.shape[0] // n_cols)
    # double minus is for upside down floor division to get ceiling division

    fig, axs = plt.subplots(n_rows,n_cols,figsize=figsize)

    if n_rows * n_cols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()
    for image, label, ax in itertools.zip_longest(images, labels, axs):
        if image is not None:
            ax.imshow(image)
        if label is not None:
            ax.set_title(label)
        ax.axis('off')
    fig.subplots_adjust(hspace = 0)
    fig.tight_layout()
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

    if old in [LAYOUT_NHWC, LAYOUT_NCHW] and x.ndim!=4:
        raise ValueError('style is LAYOUT_NP but dimensions are not 4')

    if old == LAYOUT_NP:
        if x.ndim!=2:
            raise ValueError('style is LAYOUT_NP but dimensions are not 2')
        h = kwargs.get('h')
        w = kwargs.get('w')
        c = kwargs.get('c')
        if new == LAYOUT_NHWC:
            x_ = x.reshape([-1, h, w, c])
        elif new == LAYOUT_NCHW:
            x_ = x.reshape([-1, c, h, w])
        elif new == LAYOUT_NHW:
            x_ = x.reshape([-1, h, w])
    elif new == LAYOUT_NP:
        if old == LAYOUT_NHWC:
            h = x.shape[1]
            w = x.shape[2]
            c = x.shape[3]
        elif old == LAYOUT_NCHW:
            h = x.shape[2]
            w = x.shape[3]
            c = x.shape[1]
        elif old == LAYOUT_NHW:
            if x.ndim!=3:
                raise ValueError('style is LAYOUT_NHW but dimensions are not 3')
            h = x.shape[1]
            w = x.shape[2]
            c = 1
        p = h * w * c
        x_ = np.reshape(x, [-1, p])
    elif all(layouts in [LAYOUT_NCHW, LAYOUT_NHWC] for layouts in
             [old, new]):
        new = [old.index(char) for char in new]
        x_ = np.transpose(x, new)
    else:
        raise ValueError("Can't convert from {} to {}".format(old, new))
    if x.shape[0] != x_.shape[0]:
        raise ValueError(
            "Inavalid conversion from {} with shape {} to {} with shape {}".format(
                old, x.shape, new, x_.shape))
    return x_

def load_image(image_location, cv2_flag = cv2.IMREAD_UNCHANGED):
    """ Load an image, returns numpy array """
    # if the path appears to be an URL
    if urlparse(image_location).scheme in ('http', 'https'):
        # set up the byte stream
        img_stream = np.asarray(bytearray(request.urlopen(image_location).read()))
        # and read in as PIL image
        img = cv2.imdecode(img_stream, cv2_flag)
    else:
        # else use it as local file path
        img = cv2.imread(image_location, cv2_flag)
    if img.shape[2]==3:
        img = bgr2rgb(img)

    return img

#TODO: Add layout_flag and cv2_imread_flag

def load_images(images_location, start=0, stop=100, cv2_flag = cv2.IMREAD_UNCHANGED):
    if tarfile.is_tarfile(images_location):
        images=[]
        tar = tarfile.open(images_location)
        i = 0
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f:
                if i >= start:
                    img = np.frombuffer(f.read(), dtype=np.uint8)
                    img = cv2.imdecode(img, cv2_flag)
                    if img.shape[2]==3:
                        img = bgr2rgb(img)
                    images.append(img)
                i+=1
            if i==stop:
                break
        images = np.array(images)
    else:
        images = np.array([cv2.imread(str(i), cv2_flag) for i in images_location[start:stop]])
    return images

def bgr2rgb(img):
    return img[...,::-1] # cv2.cvtColor(img, cv2.COLOR_BGR2RGB)