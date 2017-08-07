from sklearn.preprocessing import StandardScaler as skpp_StandardScaler
import pandas as pd
import numpy as np

class Dataset(object):

    def __init__(self,data=None):
        self.data = data
        self.init_part()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data=data
        self.mldata = None  # if the data is refreshed, also init mldata to none


    # this is where we store columnar data for ML
    @property
    def mldata(self):
        return self._mldata

    @mldata.setter
    def mldata(self, data=None):
        if data is None:
            self._mldata=None
        else:
            if isinstance(data,pd.DataFrame) or isinstance(data,pd.Series):
                self._mldata = data.values
            else:
                self._mldata = data
            self._mldata = self._mldata.astype(np.float32)
            if self._mldata.ndim==1:
                self._mldata = self._mldata.reshape(-1,1)
        self.scaler=None

    part_all = ['train','valid','test']

    @property
    def part_list(self):
        XY_list=[]
        for part in Dataset.part_all:
            if self.part[part] is not None:
                XY_list.append(part)
        return XY_list

    def check_part_list(self,parts = None):
        if parts is not None:
            for part in parts:
                if self.part[part] is None:
                    return False
        return True

    @property
    def X_list(self):
        X_list=[]
        for part in Dataset.part_all:
            if self.part[part] is not None:
                X_list.append('X_'+part)
        return X_list

    @property
    def Y_list(self):
        Y_list=[]
        for part in Dataset.part_all:
            if self.part[part] is not None:
                Y_list.append('Y_'+part)
        return Y_list

    def part_print(self):
        for k, v in self.part.items():
            print(k, 'None' if v is None else v.shape)

    def init_part(self):
        self.part = {
            'X'        : None,
            'Y'        : None,
            'X_train'  : None,
            'Y_train'  : None,
            'X_valid'  : None,
            'Y_valid'  : None,
            'X_test'   : None,
            'Y_test'   : None,
            'train'    : None,
            'test'     : None,
            'valid'    : None,
        }

    @property
    def X_train(self):
        return self.part['X_train']

    @property
    def X_valid(self):
        return self.part['X_valid']

    @property
    def X_test(self):
        return self.part['X_test']

    @property
    def Y_train(self):
        return self.part['Y_train']

    @property
    def Y_valid(self):
        return self.part['Y_valid']

    @property
    def Y_test(self):
        return self.part['Y_test']

    @property
    def train(self):
        return self.part['train']

    @property
    def valid(self):
        return self.part['valid']

    @property
    def test(self):
        return self.part['test']

    def StandardizeX(self):
        X_list = self.X_list if self.part['X'] is None else self.X_list.append('X')

        if X_list:
            self.scaler=skpp_StandardScaler(copy=False)
            self.part[X_list[0]] = self.scaler.fit_transform(self.part[X_list[0]])
            for part in X_list[1:]:
                if self.part[part] is not None:
                    self.part[part] = self.scaler.transform(self.part[part])
        else:
            self.scaler=None

    def StandardizeInverseX(self,data=None):
        if data is None and self.scaler is not None:
            X_list = self.X_list if self.part['X'] is None else self.X_list.append('X')
            if X_list:
                for part in X_list:
                    if self.part[part] is not None:
                        self.part[part] = self.scaler.inverse_transform(self.part[part])
        else:
            return self.scaler.inverse_transform(data, copy=True)
