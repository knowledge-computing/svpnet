from __future__ import print_function
import torch.utils.data as data

import os
import numpy as np
import random
import torch


class SST(data.Dataset):
    
    def __init__(
        self,
        data_path,
        mode='train',
        seq_len=4,
        horizon=6,
        length=10000
    ):
        
        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        self.length = length 
        
        path = os.path.join(data_path, f'sst_train.npz')
        if not os.path.exists(path):
            raise ValueError(f'sst_train.npz does not exist.')
            
        dataset = np.load(path, allow_pickle=True)
        self.daily_zone_mean = dataset['daily_zone_mean']
        self.daily_zone_std = dataset['daily_zone_std']
        self.zone_mean = dataset['zone_mean']
        self.zone_std = dataset['zone_std']
        self.min_value = dataset['min_value']
        self.max_value = dataset['max_value']
        self.data = dataset['data']
        
        if mode == 'train':
            self.samples = dataset['samples']
            print(self.samples.shape)
        elif mode == 'val':
            dataset = np.load(os.path.join(data_path, f'sst_test.npz'))
            self.data = dataset['data']
            self.samples = dataset['samples'][:length]
        elif mode == 'test':
            dataset = np.load(os.path.join(data_path, f'sst_test.npz'))
            self.data = dataset['data']
            self.samples = dataset['samples'][length: length * 2]
        
    def __getitem__(self, index):
        
        if self.mode == 'train':
            index = random.choice(range(len(self.samples)))
            
        year, day, region = self.samples[index]
        
        input_seq = self.data[year, day: day + self.seq_len, region]
        output_seq = self.data[year, day + self.seq_len: day + self.seq_len + self.horizon, region]

        input_seq = torch.from_numpy(input_seq).contiguous().float()        
        output_seq = torch.from_numpy(output_seq).contiguous().float()
        return input_seq, output_seq

    def get_meta(self, index):
        return self.samples[index]
    
    def __len__(self):
        return self.length

    
def load_dataset(data_path, seq_len, horizon):
    train_dataset = SST(data_path=data_path, 
                        mode='train', 
                        seq_len=seq_len, 
                        horizon=horizon,
                        length=10000)
    val_dataset = SST(data_path=data_path, 
                      mode='val', 
                      seq_len=seq_len, 
                      horizon=horizon,
                      length=4000)
    test_dataset = SST(data_path=data_path, 
                       mode='test', 
                       seq_len=seq_len, 
                       horizon=horizon,
                       length=4000)    
    return train_dataset, val_dataset, test_dataset
    
    
def load_test_dataset(data_path, seq_len, horizon):
    test_dataset = SST(data_path=data_path, 
                       mode='test', 
                       seq_len=seq_len, 
                       horizon=horizon,
                       length=4000)    
    return test_dataset
    