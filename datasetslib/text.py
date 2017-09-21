from .dataset import Dataset

import numpy as np
import collections

class TextDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)

    def _read_words(self,f):
        #with tf.gfile.GFile(filename, "r") as f:
        #with open(filename,'r') as f:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

    def build_word2id(self, filehandle):
        data = self._read_words(filehandle)

        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        words, _ = list(zip(*count_pairs))
        word2id = dict(zip(words, range(len(words))))

        return word2id

    def build_file2id(self, filehandle, word2id):
        data = self._read_words(filehandle)
        return [word2id[word] for word in data if word in word2id]


