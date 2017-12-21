import numpy as np
import imageio

def image_layout(x, old, new):
    new = [old.index(char) for char in new]
    return np.transpose(x,new)

def image_np2nhwc(x,h,w,c):
    return np.reshape(x,[-1,h,w,c])

def image_to2np(x,h,w,c):
    return np.reshape(x,[-1,h*w*c])

def load_images(x):
    images = np.array([imageio.imread(i) for i in x])
    return images

def one_hot(y,n_classes=0):
    if n_classes<2:
        n_classes = np.max(y)+1
    assert n_classes>1, 'Number of classes can not be less than 2'
    return np.eye(n_classes)[y]

def argmax(x):
    return np.argmax(x,axis=1)

# unit_axis =1 means one column and unit_axis = 0 means one row
def to2d(x,unit_axis=1):
    if unit_axis==1: # one column
        col = 1
        row = -1
    else:
        col = -1
        row = 1
    return np.reshape(x,[row,col])