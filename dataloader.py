import bz2
import pickle
import pickle as cPickle
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
import pickle
import os
import numpy as np


def load_data(user_train, user_test = None, evaluate = None):
    os.getcwd()
    datasetT = decompress_pickle(f'data/dataT.pbz2')
    if evaluate is not None:
        datasetE = decompress_pickle(f'data/dataE.pbz2') 
        #has to be completed
    if user_test is not None:
        data_test = { your_key: datasetT[str(your_key)]['data'] for your_key in user_test }
        label_test = { your_key: datasetT[str(your_key)]['label'] for your_key in user_test }

        data_train = { your_key: datasetT[str(your_key)]['data'] for your_key in user_train }
        label_train = { your_key: datasetT[str(your_key)]['label'] for your_key in user_train }
    
    else:
        data_train = { your_key: datasetT[str(your_key)]['data'] for your_key in user_train }
        label_train = { your_key: datasetT[str(your_key)]['label'] for your_key in user_train }
    
    if user_test is not None:
        return data_train, data_test, label_train, label_test
    
    else:
        return data_train, label_train


# Class for loading data
class ImageTensorDatasetMultiEpoch(data.Dataset):

    def __init__(self, data_input, user_ids, filter_seq, label):
        
        self.user_id, self.idx, self.idxRoll, self.segmentID = [], [], [], []

        for i, user in enumerate(user_ids):
            self.idx = np.concatenate((self.idx, np.arange(len(label[user]))), axis = 0).astype(int)
            self.user_id = np.concatenate((self.user_id, np.multiply(user, np.ones(len(label[user])))), axis = 0).astype(int)
            self.idxRoll = np.stack([np.roll(np.arange(len(label[user])), i, axis = 0) for i in range(filter_seq, -1, -1)], axis = 1)
            if i == 0:
                self.image_ix = self.idxRoll
            else:
                self.image_ix = np.concatenate((self.image_ix, self.idxRoll), axis = 0)
        
        self.segmentID = np.stack([np.roll(np.arange(len(self.user_id)), i, axis = 0) for i in range(filter_seq, -1, -1)], axis = 1)

        self.labels = label
        self.EEG_data = data_input


    def __len__(self):
        return len(self.image_ix)
    
    def __getitem__(self, key):
        dataMatrix = self.EEG_data[self.user_id[key]][self.image_ix[key, :]]
        labelsData = self.labels[self.user_id[key]][self.image_ix[key, :]]
        
        return dataMatrix, labelsData, self.segmentID[key]


# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data

# Pickle a file and then compress it into a file with extension 
def compress_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f: 
        cPickle.dump(data, f)