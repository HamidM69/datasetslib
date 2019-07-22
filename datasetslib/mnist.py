import os
import gzip
import tarfile

import numpy as np

from .image import ImageDataset
from . import dsroot
from .util import util
from .util import imutil

try:
    # Python 2
    from itertools import izip
except ImportError:
    # Python 3
    izip = zip
# from scipy.misc import imsave
import cv2


class MNIST(ImageDataset):

    def __init__(self):
        super().__init__()
        self.dataset_name = 'mnist'
        self.source_url = 'http://yann.lecun.com/exdb/mnist/'
        self.source_files = ['train-images-idx3-ubyte.gz',
                             'train-labels-idx1-ubyte.gz',
                             't10k-images-idx3-ubyte.gz',
                             't10k-labels-idx1-ubyte.gz']
        self.dataset_home = os.path.join(dsroot, self.dataset_name)

        self.height = 28
        self.width = 28
        self.depth = 1

        self._x_layout_file = imutil.LAYOUT_NHW

        self.n_features = self.height * self.width * self.depth
        self.n_classes = 10

    def load_data(self, force_download=False, shuffle=True, x_as_image=False,
                  x_layout=None):

        n_train = 60000
        n_test = 10000
        if x_layout is not None:
            self._x_layout = x_layout

        self.downloaded_files = util.download_dataset(
            source_url=self.source_url,
            source_files=self.source_files,
            dest_dir=self.dataset_home,
            force_download=force_download,
            force_extract=False)

        modified_files = False
        print('Extracting and rearchiving as jpg files...')

        for part in ['train', 'test']:
            tt_folder = os.path.join(self.dataset_home, part)
            if not os.path.isdir(tt_folder):
                os.makedirs(tt_folder)
            for i in range(self.n_classes):
                class_folder = os.path.join(tt_folder, str(i))
                if not os.path.isdir(class_folder):
                    os.makedirs(class_folder)

            print(os.path.join(self.dataset_home, self.downloaded_files[0]))

            # Extract it into np arrays.
            if part == 'train':
                data = self.read_data(filename=os.path.join(self.dataset_home,
                                                            self.downloaded_files[
                                                                0]),
                                      n_images=n_train
                                      )
                labels = self.read_labels(
                    filename=os.path.join(self.dataset_home,
                                          self.downloaded_files[1]),
                    num=n_train
                )
            else:
                data = self.read_data(filename=os.path.join(self.dataset_home,
                                                            self.downloaded_files[
                                                                2]),
                                      n_images=n_test
                                      )
                labels = self.read_labels(
                    filename=os.path.join(self.dataset_home,
                                          self.downloaded_files[3]),
                    num=n_test
                )
            print('Saving ', part)
            for i in range(len(data)):
                image_path = os.path.join(tt_folder, str(labels[i]),
                                          '{}.jpg'.format(i))
                if not os.path.isfile(image_path):
                    cv2.imwrite(image_path, data[i][:, 0])
                    modified_files = True

            if modified_files:
                print('Zipping ', part)
                with tarfile.open(
                        os.path.join(self.dataset_home, 'mnist.tar.gz'),
                        'w:gz') as tar:
                    tar.add(tt_folder, arcname=part)
            else:
                print('Zip file not modified')

        print('Loading in x and y... start')
        x_train_files = []
        y_train = []
        x_test_files = []
        y_test = []

        for i in range(self.n_classes):
            for part,x_part, y_part in zip(['train','test'],[x_train_files,x_test_files],[y_train,y_test]):
                ifolder = os.path.join(self.dataset_home, part, str(i))
                files = [name for name in os.listdir(ifolder) if
                         name.endswith('.jpg')]
                for f in files:
                    x_part.append(os.path.join(ifolder, f))
                    y_part.append(i)

        if shuffle:
            x_train_files, y_train = self.shuffle_xy(x_train_files, y_train)

        if x_as_image:
            x_train = self.load_images(x_train_files)
            x_test = self.load_images(x_test_files)
        else:
            x_train = x_train_files
            x_test = x_test_files
        self._x_as_image = x_as_image

        # no need to make onehot here as next batch returns as onehot
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        self.part['x_train'] = x_train
        self.part['y_train'] = y_train

        self.part['x_test'] = x_test
        self.part['y_test'] = y_test

        print('Loading in x and y... done')
        return x_train, y_train, x_test, y_test

    def read_data(self, filename, n_images):
        """Extract the #num images into a 4D tensor [image index, y, x, channels].
        """
        print('Reading from ', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(16)
            buf = bytestream.read(self.height * self.width * n_images)
            data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
            data = data.reshape(n_images, self.height * self.width, 1)
            return data

    def read_labels(self, filename, num):
        """Extract the labels into a vector of int64 label IDs."""
        print('Reading from ', filename)
        with gzip.open(filename) as bytestream:
            bytestream.read(8)
            buf = bytestream.read(1 * num)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.uint8)
        return labels
