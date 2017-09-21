import os
import re
import sys
from six.moves import cPickle
import tarfile

import numpy as np

from .images import ImagesDataset
from . import datasets_root
from . import util
try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip



class cifar10(ImagesDataset):
    def __init__(self):
        ImagesDataset.__init__(self)
        self.dataset_name='cifar10'
        self.source_url='http://www.cs.toronto.edu/~kriz/'
        self.source_files=['cifar-10-python.tar.gz']
        self.dataset_home=os.path.join(datasets_root,self.dataset_name)

        self.height=32
        self.width=32
        self.depth=3

        self.x_shape = 'NCHW' ## other alternates are NHW and NHCW
        self.n_classes = 10

    def load_data(self,force=False,x_shape=None,one_hot=True):
        self.downloaded_files=util.download_dataset(source_url=self.source_url,
                                                    source_files=self.source_files,
                                                    dest_dir = self.dataset_home,
                                                    force=force,
                                                    extract=False)

        n_train = 50000
        n_test = 10000

        print('Extracting ',self.downloaded_files[0])
        x_train = np.zeros((n_train, 3, 32, 32), dtype=np.uint8)
        y_train = np.zeros((n_train,), dtype=np.uint8)

        with tarfile.open(self.downloaded_files[0]) as archfile:
            for i in range(1, 6):
                pfile='cifar-10-batches-py/data_batch_' + str(i)
                f = archfile.extractfile(pfile)
                #self.pload(f)

                #fpath = os.path.join(self.dataset_home, 'data_batch_' + str(i))
                #print(fpath)
                data, labels = cifar10.load_pfile(f=f)
                x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
                y_train[(i - 1) * 10000: i * 10000] = labels

            pfile='cifar-10-batches-py/test_batch'
            f = archfile.extractfile(pfile)
            x_test, y_test = cifar10.load_pfile(f=f)

        y_train = np.reshape(y_train,len(y_train),1)
        y_test = np.reshape(y_test,len(y_test),1)

        if one_hot:
            y_train = util.np_one_hot(n_classes=self.n_classes,y=y_train)
            y_test = util.np_one_hot(n_classes=self.n_classes,y=y_test)

        if x_shape is not None:
            self.x_shape = x_shape
        if self.x_shape == 'NHWC':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)
        self.part['X_train']=x_train
        self.part['X_test']=x_test

        self.part['Y_train'] = y_train
        self.part['Y_test'] = y_test

        return x_train, y_train, x_test, y_test

    @staticmethod
    def load_pfile(f):
        #f = open(fpath, 'rb')
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # decode utf8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
        #        f.close()
        data = d['data']
        labels = d['labels']

        # data is in shape NP and can be converted to NCHW
        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels