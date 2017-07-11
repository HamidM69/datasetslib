from .dataset import Dataset
import numpy as np
from pandas import DataFrame
from pandas import concat
from datetime import timedelta
from . import util

# convert an array of values into a dataset matrix
# x_block is the number of elements to be considered in one X row
# y_block is the number of elements to be considered in one Y row
# x_step is the delta-T from first elements of previous row for next X row,
#        i.e. for x_step = 2, if the first row has t1,t2,t3 then second row has t3,t4,t5
# y_step is the delta-T for first element of Y from last element of x
#        i.e. for y_step =1, if the x has t1,t2,t3 then y has t4


#    if hasattr(X, "iloc"):
#        # Pandas Dataframes and Series
#        try:
#            return X.iloc[indices]
#        except ValueError:
#            # Cython typed memoryviews internally used in pandas do not support
#            # readonly buffers.
#            warnings.warn("Copying input dataframe for slicing.",
#                          DataConversionWarning)
#            return X.copy().iloc[indices]
#    elif hasattr(X, "shape"):
#        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
#                                           indices.dtype.kind == 'i'):
#            # This is often substantially faster than X[indices]
#            return X.take(indices, axis=0)
#        else:
#            return X[indices]
#    else:
#        return [X[idx] for idx in indices]

# n_x and n_y are number of x and y timesteps respectively
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


# def mvts_to_xy(*tslist, n_x=1, n_y=1):
#     n_ts = len(tslist)
#     if n_ts == 0:
#         raise ValueError('At least one timeseries required as input')
#
#     #TODO: Validation of other options
#
#     result = []
#
#     for ts in tslist:
#         n_vars = 1 if type(ts) is list else ts.shape[1]
#         len_ts = len(ts) if type(ts) is list else ts.shape[0]
#         n_rows = len_ts - n_x - n_y + 1
#
#         dataX=np.empty(shape=(n_rows, n_vars * n_x),dtype=np.float32)
#         dataY=np.empty(shape=(n_rows, n_vars * n_y),dtype=np.float32)
#         df = DataFrame(ts)
#         x_cols, y_cols, names = list(), list(), list()
#
#         # input sequence x (t-n, ... t-1)
#         for i in range(n_x, 0, -1):
#             dataX[:,(n_x-i)*n_vars:(n_x-i)*n_vars+n_vars]=shift(ts,i)[n_x:len_ts-n_y+1]
#
#         # forecast sequence (t, t+1, ... t+n)
#         for i in range(0, n_y):
#             #y_cols.append(shift(ts,-i))
#             dataY[:,i*n_vars:i*n_vars+n_vars]=shift(ts,-i)[n_x:len_ts-n_y+1]
#
#         # put it all together
#         #x_agg = concat(x_cols, axis=1).dropna(inplace=True)
#         #y_agg = concat(y_cols, axis=1).dropna(inplace=True)
#
#         #dataX = np.array(x_cols,dtype=np.float32)
#         #dataY = np.array(y_cols,dtype=np.float32)
#
#         result.append(dataX)
#         result.append(dataY)
#     return result

# def ts_to_xy(*tslist, x_block=1, y_block=1, x_step=None, y_step=None):
#     n_ts = len(tslist)
#     if n_ts == 0:
#         raise ValueError('At least one array required as input')
#
#     if x_step is None:
#         x_step = 1
#     if y_step is None:
#         y_step = 1
#
# #TODO: Validation of other options
#
#     result = []
#     for ts in tslist:
#         dataX, dataY = [], []
#         for i in range(0,len(ts)-(x_block-1)-((y_step-1)+y_block), x_step):
#             tx_end = i + x_block
#             dataX.append(ts[i : tx_end,0])
#             ty_begin = tx_end + y_step - 1
#             dataY.append(ts[ty_begin : (ty_begin+y_block),0])
#
#         dataX = np.array(dataX,dtype=np.float32)
#         dataY = np.array(dataY,dtype=np.float32)
#         if(y_block==1):
#             dataY = dataY.ravel()
#         result.append(dataX)
#         result.append(dataY)
#     return result


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

def shift(arr, num, fill_value=np.nan):
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

# n_x and n_y are number of x and y timesteps respectively
# def mvts_to_xy(*tslist, n_x=1, n_y=1, x_idx=None, y_idx=None):
#     n_ts = len(tslist)
#     if n_ts == 0:
#         raise ValueError('At least one timeseries required as input')
#
#     #TODO: Validation of other options
#
#     result = []
#
#
#
#     for ts in tslist:
#         ts_cols = 1 if ts.ndim==1 else ts.shape[1]
#         if x_idx is None:
#             x_idx = range(0,ts_cols)
#         if y_idx is None:
#             y_idx = range(0,ts_cols)
#
#         n_x_vars = len(x_idx)
#         n_y_vars = len(y_idx)
#
#         ts_rows = ts.shape[0]
#         n_rows = ts_rows - n_x - n_y + 1
#
#         dataX=np.empty(shape=(n_rows, n_x_vars * n_x),dtype=np.float32)
#         dataY=np.empty(shape=(n_rows, n_y_vars * n_y),dtype=np.float32)
#         x_cols, y_cols, names = list(), list(), list()
#
#         # input sequence x (t-n, ... t-1)
#         from_col = 0
#         for i in range(n_x, 0, -1):
#             dataX[:,from_col:from_col+n_x_vars]=shift(ts[:,x_idx],i)[n_x:ts_rows-n_y+1]
#             from_col = from_col+n_x_vars
#
#         # forecast sequence (t, t+1, ... t+n)
#         from_col = 0
#         for i in range(0, n_y):
#             #y_cols.append(shift(ts,-i))
#             dataY[:,from_col:from_col+n_y_vars]=shift(ts[:,y_idx],-i)[n_x:ts_rows-n_y+1]
#             from_col = from_col + n_y_vars
#
#         # put it all together
#         #x_agg = concat(x_cols, axis=1).dropna(inplace=True)
#         #y_agg = concat(y_cols, axis=1).dropna(inplace=True)
#
#         #dataX = np.array(x_cols,dtype=np.float32)
#         #dataY = np.array(y_cols,dtype=np.float32)
#
#         result.append(dataX)
#         result.append(dataY)
#     return result


#N = DATA.shape[0]

#ratio = (ratio*N).astype(np.int32)

#ind = np.random.permutation(N)
#X_train = DATA[ind[:ratio[0]],1:]
#X_val = DATA[ind[ratio[0]:ratio[1]],1:]
#X_test = DATA[ind[ratio[1]:],1:]
# Targets have labels 1-indexed. We subtract one for 0-indexed
#y_train = DATA[ind[:ratio[0]],0]-1
#y_val = DATA[ind[ratio[0]:ratio[1]],0]-1
#y_test = DATA[ind[ratio[1]:],0]-1

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

def next_weekday(d, weekday, next=True):
    days_ahead = weekday - d.weekday()
    if next:
        if days_ahead < 0: # Target day already happened this week
            days_ahead += 7
    else:
        if days_ahead > 0: # Target day already happened this week
            days_ahead -= 7
    return d + timedelta(days_ahead)


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

from . import datasets_root
import glob
import pandas as pd
import numpy as np
from datetime import datetime

class TimeSeriesDataset(Dataset):
    def __init__(self, data=None):
        Dataset.__init__(self, data)



    @Dataset.mldata.setter
    def mldata(self, data=None):
        if data is None:
            self._mldata=None
            self.init_part()
        else:
            if isinstance(data,pd.DataFrame) or isinstance(data,pd.Series):
                self._mldata = data.values
            else:
                self._mldata = data
            if self._mldata.ndim==1:
                self._mldata = self._mldata.reshape(-1,1)
            self._mldata = self._mldata.astype(np.float32)


    # this function assumes there is no missing data
    # dataframe cols in col_idx are numeric
    def train_test_split_dayofweek(self, train_days:list=[0,1,2,3],test_days:list=[4], X_weeks=1,Y_weeks=1, col_names=None):

        if self._data is None:
            raise ValueError('No timeseries found in _data')

        all_days = set(train_days) | set(test_days)
        #all_days.sort()
        #print(all_days)
        if col_names is None:
            col_names = self._data.columns.tolist()

        # get the days falling in the range
        min_date = self._data.index.min()
        max_date = self._data.index.max()
        first_day = next_weekday(min_date, min(all_days)) # 0 = Monday, 1=Tuesday, 2=Wednesday...
        last_day = next_weekday(max_date,max(all_days),False)
        #print('min_date:',min_date)
        #print('max_date:',max_date)
        #print('First Monday:',first_monday)
        #print('Last Friday:',last_friday)
        #print(col_names)
        #print(self._data.index.dayofweek in all_days)
        day_df = self._data.loc[(self._data.index.dayofweek.isin(all_days)) & (self._data.index >= first_day) & (self._data.index <= last_day), col_names]

        self.mldata = day_df.values.astype(np.float32)
        self.mldata_col_names = col_names

        self.part['train'] = day_df.loc[day_df.index.dayofweek.isin(train_days)].values.astype(np.float32)
        self.part['test'] =  day_df.loc[day_df.index.dayofweek.isin(test_days)].values.astype(np.float32)
        self.part['valid'] = None

        #print(self.mldata_cols)
        #print(self.part['train'].dtype)
        #print('train shape:',self.part['train'].shape)
        results=[]
        results.append(self.part['train'])
        results.append(self.part['test'])

        #print(self._mldata.shape)


        #print(self.part['test'].shape)

        # now split into train and test
        return results


    # in case of timeseries, always split test train first
    def train_test_split(self, train_size=0.75, val_size=0):

        results=[]
        if self._mldata is None:
                raise ValueError('No timeseries found')

        if train_size > 1 or train_size <=0:
            raise ValueError('train_size has to be between 0 and 1')

        if val_size >= 1 or val_size < 0:
            raise ValueError('val_size has to be between 0 and 1')

        N = self._mldata.shape[0]

        train_size = int(N * train_size)
        val_size = int(N * val_size)
        test_size = N - train_size - val_size

        if(train_size>0):
            self.part['train'] = self._mldata[0:train_size]
            results.append(self.part['train'])
        else:
            self.part['train'] = None

        if(val_size>0):
            self.part['valid'] = self._mldata[train_size:train_size+val_size]
            results.append(self.part['valid'])
        else:
            self.part['valid'] = None

        if(test_size>0):
            self.part['test'] =  self._mldata[train_size+val_size:N]
            results.append(self.part['test'])
        else:
            self.part['test'] = None
        return results


    # n_tx and n_ty are number of x and y timesteps respectively,
    # h is prediction horizon for direct strategy
    # returns input series : {t-n_tx,...,t}, output series : {t+h,...,t+h+n_ty}
    # x_idx is the list of columns that would be used as input or feature
    # y_idx is the list of columns that would be used as output or target
    def mvts_to_xy(self, n_tx=1, n_ty=1, x_idx=None, y_idx=None, h=1, only_parts=None):
        if self._mldata is None:
            raise ValueError('No timeseries found')

        ts_cols = 1 if self._mldata.ndim==1 else self._mldata.shape[1]
        if x_idx is None:
            x_idx = range(0,ts_cols)
        if y_idx is None:
            y_idx = range(0,ts_cols)
        self.x_idx = x_idx
        self.y_idx = y_idx
        self.y_cols_x_idx = [x_idx.index(i) for i in y_idx]

        n_x_vars = len(x_idx)
        n_y_vars = len(y_idx)

        if only_parts is not None:
            if self.check_part_list(only_parts):
                ts_list = only_parts
            else:
                ts_list = []
                raise ValueError('the part you have asked to split is not available')
        else:
            ts_list = self.part_list
            if not ts_list:
                raise ValueError('Timeseries has not been split into train, valid, test. Run train_test_split method first')


        #TODO: Validation of other options

        result = []

        # as of now we are only converting the training and test set
        # train set is to be converted based on strategy
        # for single step ahead prediction : input series : {t-n_tx,...,t}, output series : {t+1}
        # for multi step ahead :
        #   iterative : input series : {t-n_tx,...,t}, output series : {t+1} and columns of out_vec in input series
        #   direct : input series : {t-n_tx,...,t}, output series : {t+h}
        #   MIMO : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}
        # test set is always going to be : input series : {t-n_tx,...,t}, output series : {t+1,...,t+n_ty}


        for ts_part in Dataset.part_all:
            if ts_part not in ts_list:
#                self.part['X_'+ts_part]=None
#                self.part['Y_'+ts_part]=None
                pass
            else:
                ts = self.part[ts_part]
                #print(ts)

                ts_rows = ts.shape[0]
                #print(ts_rows)
                n_rows = ts_rows - n_tx - (n_ty - 1) - (h - 1)
                #print(n_rows)
                dataX=np.empty(shape=(n_rows, n_x_vars * n_tx), dtype=np.float32)
                dataY=np.empty(shape=(n_rows, n_y_vars * n_ty), dtype=np.float32)

                #print(dataX.shape)
                #print(dataY.shape)

                #print(ts.shape)
                #print(x_idx)

                # input sequence x (t-n_tx, ... t)

                from_col = 0
                #print(from_col)
                #print(from_col+n_x_vars)

                for i in range(n_tx, 0, -1):
                    dataX[:,from_col:from_col+n_x_vars]= util.shift(ts[:,x_idx],i)[n_tx:n_rows+n_tx]
                    from_col = from_col+n_x_vars

                # forecast sequence (t+h, ... t+h+n_ty)
                from_col = 0
                for i in range(0, n_ty):
                    #y_cols.append(shift(ts,-i))
                    dataY[:,from_col:from_col+n_y_vars]= util.shift(ts[:,y_idx],-(i+h-1))[n_tx:n_rows+n_tx]
                    from_col = from_col + n_y_vars

                result.append(dataX)
                result.append(dataY)

                self.part['X_'+ts_part]=dataX
                self.part['Y_'+ts_part]=dataY

        return result


