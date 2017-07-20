import numpy as np
import six
import sys
import datetime
import gc
import time

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
    print('obj size:{}'.format(sys.getsizeof(obj)))
    for attr in vars(obj):
        print('  .{} type:{}'.format(attr,type(attr)))