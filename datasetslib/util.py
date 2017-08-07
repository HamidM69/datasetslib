import numpy as np
import six
import sys
import datetime
import gc
import time
from datetime import timedelta

def df_to_sarray(df):
    """
    Convert a pandas DataFrame object to a numpy structured array.
    This is functionally equivalent to but more efficient than
    np.array(df.to_array())

    :param df: the data frame to convert
    :return: a numpy structured array representation of df
    """

    vals = df.values.astype(np.float32)
    #print('vals shape:',vals.shape)
    cols = df.columns
    #print[cols]
    if six.PY2:  # python 2 needs .encode() but 3 does not
        types = [(cols[i].encode(), df[k].dtype.type) for (i, k) in enumerate(cols)]
    else:
        types = [(cols[i], df[k].dtype.type) for (i, k) in enumerate(cols)]

    #if six.PY2:  # python 2 needs .encode() but 3 does not
    #    types = [(cols[i].encode(), np.dtype(np.float32).type) for (i, k) in enumerate(cols)]
    #else:
    #    types = [(cols[i], np.dtype(np.float32).type) for (i, k) in enumerate(cols)]

    dtype = np.dtype(types)

    #print(dtype)
    z = np.zeros((vals.shape[0],), dtype)
    print('z shape:',z.shape)
    for (i, k) in enumerate(z.dtype.names):
        z[k] = vals[:, i]
    return z

def shift(arr, num, fill_value=np.nan):
    #print(arr)
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result = arr
    return result

def to2d(arr):
    if arr.ndim==1:
        arr = arr.reshape(-1,1)
    return arr

sflush = sys.stdout.flush

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

class ExpTimer:
    def start(self):
        gc.collect()
        gc.disable()
        self.start_time = time.process_time()
    def stop(self):
        self.stop_time = time.process_time()
        gc.enable()
        gc.collect()
    @property
    def elapsedTime(self):
        return self.stop_time - self.start_time

def objvars(obj):
    print('obj size:{0}'.format(sys.getsizeof(obj)))
    for attr in vars(obj):
        print('  .{0} type:{1}'.format(attr,type(attr)))


def next_weekday(d, weekday, next=True):
    days_ahead = weekday - d.weekday()
    if next:
        if days_ahead < 0: # Target day already happened this week
            days_ahead += 7
    else:
        if days_ahead > 0: # Target day already happened this week
            days_ahead -= 7
    return d + timedelta(days_ahead)

def mvts_to_xy(*tslist, n_x=1, n_y=1, x_idx=None, y_idx=None):
    n_ts = len(tslist)
    if n_ts == 0:
        raise ValueError('At least one timeseries required as input')

    #TODO: Validation of other options

    result = []

    for ts in tslist:
        ts_cols = 1 if ts.ndim==1 else ts.shape[1]
        if x_idx is None:
            x_idx = range(0,ts_cols)
        if y_idx is None:
            y_idx = range(0,ts_cols)

        n_x_vars = len(x_idx)
        n_y_vars = len(y_idx)

        ts_rows = ts.shape[0]
        n_rows = ts_rows - n_x - n_y + 1

        dataX=np.empty(shape=(n_rows, n_x_vars * n_x),dtype=np.float32)
        dataY=np.empty(shape=(n_rows, n_y_vars * n_y),dtype=np.float32)
        x_cols, y_cols, names = list(), list(), list()

        # input sequence x (t-n, ... t-1)
        from_col = 0
        for i in range(n_x, 0, -1):
            dataX[:,from_col:from_col+n_x_vars]=shift(ts[:,x_idx],i)[n_x:ts_rows-n_y+1]
            from_col = from_col+n_x_vars

        # forecast sequence (t, t+1, ... t+n)
        from_col = 0
        for i in range(0, n_y):
            #y_cols.append(shift(ts,-i))
            dataY[:,from_col:from_col+n_y_vars]=shift(ts[:,y_idx],-i)[n_x:ts_rows-n_y+1]
            from_col = from_col + n_y_vars

        # put it all together
        #x_agg = concat(x_cols, axis=1).dropna(inplace=True)
        #y_agg = concat(y_cols, axis=1).dropna(inplace=True)

        #dataX = np.array(x_cols,dtype=np.float32)
        #dataY = np.array(y_cols,dtype=np.float32)

        result.append(dataX)
        result.append(dataY)
    return result

# in case of timeseries, always split test train first
def train_test_split(timeseries, train_size=0.75, val_size=0.0):

    if train_size >= 1:
        raise ValueError('train_size has to be between 0 and 1')

    if val_size >= 1:
        raise ValueError('val_size has to be between 0 and 1')

    N = timeseries.shape[0]

    train_size = int(N * train_size)
    val_size = int(N * val_size)
    test_size = N - train_size - val_size

    if(val_size>0):
        train, val,test = timeseries[0:train_size,:], timeseries[train_size:train_size+val_size,:], timeseries[train_size+val_size:N,:]
        return train,val,test
    else:
        train, test = timeseries[0:train_size,:], timeseries[train_size:len(timeseries),:]
        return train,test

def sample_batch(*tslist,batch_size):
    """ Function to sample a batch for training"""

    n_ts = len(tslist)
    if n_ts == 0:
        raise ValueError('At least one timeseries required as input')

    #TODO: Validation of other options

    result = []
    for ts in tslist:

        N = ts.shape[0]
        N_idx = np.random.choice(N,batch_size,replace=False)
        result.append(ts[N_idx])

    return result

# def next_batch(self, batch_size, fake_data=False, shuffle=True):
#     """Return the next `batch_size` examples from this data set."""
#     if fake_data:
#         fake_image = [1] * 784
#         if self.one_hot:
#             fake_label = [1] + [0] * 9
#         else:
#             fake_label = 0
#         return [fake_image for _ in xrange(batch_size)], [
#             fake_label for _ in xrange(batch_size)
#         ]
#     start = self._index_in_epoch
#     # Shuffle for the first epoch
#     if self._epochs_completed == 0 and start == 0 and shuffle:
#         perm0 = numpy.arange(self._num_examples)
#         numpy.random.shuffle(perm0)
#         self._images = self.images[perm0]
#         self._labels = self.labels[perm0]
#     # Go to the next epoch
#     if start + batch_size > self._num_examples:
#         # Finished epoch
#         self._epochs_completed += 1
#         # Get the rest examples in this epoch
#         rest_num_examples = self._num_examples - start
#         images_rest_part = self._images[start:self._num_examples]
#         labels_rest_part = self._labels[start:self._num_examples]
#         # Shuffle the data
#         if shuffle:
#             perm = numpy.arange(self._num_examples)
#             numpy.random.shuffle(perm)
#             self._images = self.images[perm]
#             self._labels = self.labels[perm]
#         # Start next epoch
#         start = 0
#         self._index_in_epoch = batch_size - rest_num_examples
#         end = self._index_in_epoch
#         images_new_part = self._images[start:end]
#         labels_new_part = self._labels[start:end]
#         return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
#     else:
#         self._index_in_epoch += batch_size
#         end = self._index_in_epoch
#         return self._images[start:end], self._labels[start:end]

def getfunc(fname, objects=globals()):
    fn = None
    if isinstance(fname, six.string_types):
        fn = objects.get(str(fname))
    elif callable(fname):
        fn = fname
    else:
        fn = None

    if fn is None:
        raise(ValueError('No such function: {0}'.format(fname)))
    else:
        return fn
