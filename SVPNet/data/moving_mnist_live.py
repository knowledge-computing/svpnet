import gzip
import math
import numpy as np
import os
from PIL import Image
import random
import torch
import torch.utils.data as data


def load_mnist(root):
    
    image_path = os.path.join(root, 'train-images-idx3-ubyte.gz')
    label_path = os.path.join(root, 'train-labels-idx1-ubyte.gz')
    
    if not os.path.exists(image_path):
        raise ValueError('Image path does not exist')
    if not os.path.exists(label_path):
        raise ValueError('Label path does not exist')
        
    with gzip.open(image_path, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
        images = images.reshape(-1, 28, 28)
    with gzip.open(label_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
        labels = labels.reshape(-1)        
    return images, labels


def select_digits(labels, selected_digits):
    
    while True:
        ind = random.randint(0, labels.shape[0] - 1)
        if selected_digits is None:
            return ind
        else:
            if labels[ind] in selected_digits:
                return ind

        
def load_fixed_set(root, data_source):
    
    if data_source == 'moving_mnist_ori':
        path = os.path.join(root, 'mnist_test_seq.npy')
        data = np.load(path)
        data = data.transpose(1, 0, 2, 3)
        data = data[:, :, np.newaxis, ...]
        step_length = [0.1] * 20
        num_digits = 2
    else:
        path = os.path.join(root, data_source + '.npz')
        dataset = np.load(path)
        data = dataset['data']
        step_length = dataset['step_length']
        num_digits = dataset['num_digits']        
    
    assert data.shape[0] > data.shape[1], "num_frames must be the second dimension"
        
    data = data[:10000, ...]
    return data, step_length, num_digits


class MovingMNIST(data.Dataset):
    def __init__(self, root, data_source, 
                 is_train=True, 
                 seq_len=10,
                 horizon=10,
                 selected_digits=None,
                 transform=None):
        
        super(MovingMNIST, self).__init__()

        self.data = None
        
        self.is_train = is_train
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_frames = seq_len + horizon
        self.selected_digits = selected_digits
        self.transform = transform
        
        self.test_data, self.step_length, self.num_digits = load_fixed_set(root, data_source)

        if is_train:
            self.images, self.labels = load_mnist(root)
            self.length = int(1e4)            
        else:
            self.length = min(5000, self.test_data.shape[0])

        # For generating data
        self.image_size = 64
        self.digit_size = 28

    def get_random_trajectory(self, seq_length):
        ''' Generate a random sequence of a MNIST digit '''
        canvas_size = self.image_size - self.digit_size
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            # Take a step along velocity.
            y += v_y * self.step_length[i]
            x += v_x * self.step_length[i]

            # Bounce off edges.
            if x <= 0:
                x = 0
                v_x = -v_x
            if x >= 1.0:
                x = 1.0
                v_x = -v_x
            if y <= 0:
                y = 0
                v_y = -v_y
            if y >= 1.0:
                y = 1.0
                v_y = -v_y
            start_y[i] = y
            start_x[i] = x

        # Scale to the size of the canvas.
        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)
        return start_y, start_x

    def generate_moving_mnist(self):
        '''
        Get random trajectories for the digits and generate a video.
        '''
        data = np.zeros((self.num_frames, self.image_size, self.image_size), dtype=np.float32)
        for n in range(self.num_digits):
            # Trajectory
            start_y, start_x = self.get_random_trajectory(self.num_frames)            
            ind = random.randint(0, self.images.shape[0] - 1)
            digit_image = self.images[ind]
            
            for i in range(self.num_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.digit_size
                right = left + self.digit_size
                # Draw digit
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)

        data = data[:, np.newaxis, ...]
        return data

    def __getitem__(self, idx):

        if self.is_train:
            images = self.generate_moving_mnist()
        else:
            images = self.test_data[idx]

        input_seq = images[:self.seq_len]
        output_seq = images[self.seq_len: self.num_frames]

        input_seq = torch.from_numpy(input_seq / 255.0).contiguous().float()        
        output_seq = torch.from_numpy(output_seq / 255.0).contiguous().float()
        return input_seq, output_seq

    def __len__(self):
        return self.length

    
def load_dataset(data_source, data_path, seq_len, horizon):
    train_dataset = MovingMNIST(root=data_path,
                                data_source=data_source,
                                is_train=True, 
                                seq_len=seq_len, 
                                horizon=horizon)
    test_dataset = MovingMNIST(root=data_path,
                               data_source=data_source,
                               is_train=False, 
                               seq_len=seq_len, 
                               horizon=horizon)
    return train_dataset, test_dataset, test_dataset
    
    
def load_test_dataset(data_source, data_path, seq_len, horizon):
    test_dataset = MovingMNIST(root=data_path,
                               data_source=data_source,
                               is_train=False, 
                               seq_len=seq_len, 
                               horizon=horizon)
    return test_dataset
