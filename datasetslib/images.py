# images are stored in
#  NCHW (cuDNN default)
# Transformation available for NHW and NHWC (TensorFlow default)


import os
import sys

from . import util
from .dataset import Dataset
import numpy as np


class ImagesDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)
        self.x_shape = 'NCHW' ## other alternates are NP, NHW, and NHCW

    def scaleX(self, min=0, max=255):
        for x in self.X_list:
            self.part[x] = (self.part[x].astype(np.float32) - min) / (max - min)



