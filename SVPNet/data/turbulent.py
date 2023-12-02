import os
import h5py
import random
import numpy as np

import torch
from torch.utils.data import Dataset


class Turbulent(Dataset):
    
    def __init__(
        self, 
        data_path,
        mode='train',
        seq_len=10, 
        horizon=10
    ):
        
        # data dimensions
        self.data_path = data_path
        self.mode = mode
        self.seq_len = seq_len
        self.horizon = horizon
        
        dataset = np.load(os.path.join(data_path, 'turbulent.npz'), allow_pickle=True)
        self.data = dataset['data']
        samples = dataset['samples']
        
        if mode == 'train':
            self.samples = samples[0: 6000]
        elif mode == 'val':
            self.samples = samples[6000: 7700]
        elif mode == 'test':
            self.samples = samples[7700: 9800]

        self.mid = 40        
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        
        sample = self.samples[idx]
        seq = self.data[sample[0], sample[1]: sample[1] + 100]
        input_seq = seq[self.mid - self.seq_len: self.mid]
        output_seq = seq[self.mid: self.mid + self.horizon]
        input_seq = torch.from_numpy(input_seq).contiguous().float()        
        output_seq = torch.from_numpy(output_seq).contiguous().float()
        return input_seq, output_seq
        

def load_dataset(data_path, seq_len, horizon):
    train_dataset = Turbulent(data_path=data_path, 
                              mode='train', 
                              seq_len=seq_len, 
                              horizon=horizon)
    val_dataset = Turbulent(data_path=data_path,
                            mode='val', 
                            seq_len=seq_len, 
                            horizon=horizon)
    test_dataset = Turbulent(data_path=data_path,
                          mode='test', 
                          seq_len=seq_len, 
                          horizon=horizon)
    return train_dataset, val_dataset, test_dataset


def load_test_dataset(data_path, seq_len, horizon):
    test_dataset = Turbulent(data_path=data_path,
                             mode='test', 
                             seq_len=seq_len, 
                             horizon=horizon)
    return test_dataset



