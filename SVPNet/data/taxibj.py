import os
import sys
import h5py
import random
import numpy as np

import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset



sys.path.append('../../')
from data.TaxiBJ.TaxiBJ import *


class TaxiBJ(Dataset):
    
    def __init__(
        self, 
        data_path,
        mode='train',
        len_closeness=4, # this is seq_len
        len_trend=0,  # the same time in previous 3 weeks
        len_period=0,   # the same time in previous 3 days
        len_horizon=4,         
        meta_data=False,
        meteorol_data=False,
        holiday_data=False
    ):
        
        """
        Params:
        data_path (str): path to the parent folder containing original .h5 
        catch_path (str): path to the folder containing cached preprocessed dataset 
        source_vars (list): input sequence variables
        target_vars (list): output sequence variables, by default same as source variables    
        seq_len (int): input sequence length. Default: 4
        horizon (int): output sequence length. Default: 4.
        """
    
        # data dimensions
        self.data_path = data_path
        self.mode = mode
        self.seq_len = len_closeness
        self.horizon = len_horizon
        self.days = 28  # total 48 samples within a day
        
        fname = os.path.join(data_path, 'CACHE', 'TaxiBJ_C{}_P{}_T{}.h5'.format(len_closeness, len_period, len_trend))
        dataset = load_data(data_path=data_path, 
                            len_closeness=len_closeness,
                            len_period=len_period,
                            len_trend=len_trend,
                            len_horizon=len_horizon,
                            meta_data=meta_data,
                            meteorol_data=meteorol_data,
                            holiday_data=holiday_data)
        
        if mode == 'train':
            random.seed(1234)
            total_num = dataset['XC'][: -48 * self.days + 10].shape[0]
            random_sample = random.sample([i for i in range(total_num)], 10000)
            self.X = dataset['XC'][: -48 * self.days + 10][random_sample]
            self.Y = dataset['Y'][: -48 * self.days + 10][random_sample]    
        elif mode == 'val':
            self.X = dataset['XC'][-48 * self.days + 10: ]
            self.Y = dataset['Y'][-48 * self.days + 10: ]
        elif mode == 'test':
            self.X = dataset['XC'][-48 * self.days + 10: ]
            self.Y = dataset['Y'][-48 * self.days + 10: ]
                
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.X[idx]).contiguous().float()  
        output_seq = torch.tensor(self.Y[idx]).contiguous().float()
        return input_seq, output_seq
        

def load_dataset(data_path, seq_len, horizon):
    train_dataset = TaxiBJ(data_path=data_path, 
                           mode='train', 
                           len_closeness=seq_len, 
                           len_horizon=horizon)
    val_dataset = TaxiBJ(data_path=data_path,
                         mode='val', 
                         len_closeness=seq_len, 
                         len_horizon=horizon)
    test_dataset = TaxiBJ(data_path=data_path,
                          mode='test', 
                          len_closeness=seq_len, 
                          len_horizon=horizon)
    return train_dataset, val_dataset, test_dataset


def load_test_dataset(data_path, seq_len, horizon):
    test_dataset = TaxiBJ(data_path=data_path,
                          mode='test', 
                          len_closeness=seq_len, 
                          len_horizon=horizon)
    return test_dataset



