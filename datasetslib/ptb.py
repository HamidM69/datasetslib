import os
import tarfile
import numpy as np


from . import datasets_root
from .text import TextDataset
from . import util

class PTBSimple(TextDataset):
    def __init__(self):
        TextDataset.__init__(self)
        self.dataset_name='ptb-simple'
        self.source_url='http://www.fit.vutbr.cz/~imikolov/rnnlm/'
        self.source_files=['simple-examples.tgz']
        self.dataset_home=os.path.join(datasets_root,self.dataset_name)

    def load_data(self,force=False,):
        self.downloaded_files=util.download_dataset(source_url=self.source_url,
                                                    source_files=self.source_files,
                                                    dest_dir = self.dataset_home,
                                                    force=force,
                                                    extract=False)

        trainfile ='./simple-examples/data/ptb.train.txt'
        validfile = './simple-examples/data/ptb.valid.txt'
        testfile = './simple-examples/data/ptb.test.txt'

        with tarfile.open(self.downloaded_files[0]) as archfile:
            f = archfile.extractfile(trainfile)
            word2id = self.build_word2id(filehandle=f)
            f.seek(0)
            self.part['train'] = self.build_file2id(f,word2id)

            f = archfile.extractfile(validfile)
            self.part['valid'] = self.build_file2id(f,word2id)

            f = archfile.extractfile(testfile)
            self.part['test'] = self.build_file2id(f,word2id)

            self.vocab_len = len(word2id)
        return self.part['train'], self.part['valid'], self.part['test']

    # n_tx and n_ty are number of x and y timesteps respectively,
    # h is prediction horizon for direct strategy
    # returns input series : {t-n_tx,...,t}, output series : {t+h,...,t+h+n_ty}
    # x_idx is the list of columns that would be used as input or feature
    # y_idx is the list of columns that would be used as output or target
    def mvts_to_xy(self, n_tx=1, n_ty=1, x_idx=None, y_idx=None, h=1, ts_list=None):

        if ts_list is not None:
            if not self.check_part_list(ts_list):
                raise ValueError('the part you have asked to split is not available')
        else:
            ts_list = self.part_list
            if not ts_list:
                raise ValueError('Timeseries has not been split into train, valid, test. Run one of the train_test_split method first')

        ts_cols = 1 if self.part[ts_list[0]].ndim==1 else self.part[ts_list[0]].shape[1]

        if x_idx is None:
            x_idx = range(0,ts_cols)
        if y_idx is None:
            y_idx = range(0,ts_cols)

        self.x_idx = x_idx
        self.y_idx = y_idx
        self.y_cols_x_idx = [x_idx.index(i) for i in y_idx]

        n_x_vars = len(x_idx)
        n_y_vars = len(y_idx)




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