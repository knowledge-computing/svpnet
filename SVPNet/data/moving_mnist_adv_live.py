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

        
class MovingMNIST(data.Dataset):
    def __init__(self, 
                 root, 
                 data_source, 
                 is_train=True, 
                 seq_len=10,
                 horizon=10):
        
        super(MovingMNIST, self).__init__()
        
        self.is_train = is_train
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_frames = seq_len + horizon

        path = os.path.join(root, data_source + '.npz')
        data = np.load(path)
        self.num_digits = data['num_digits']
        self.num_rotates = data['num_rotates']
        self.num_translates = data['num_translates']
        self.bounce = data.get('bounce')
        self.direcs = data.get('direcs')
        self.speeds = data.get('speeds')
        self.selected_digits = data.get('selected_digits')
        
        if is_train:
            self.images, self.labels = load_mnist(root)
            self.length = int(1e4)            
        else:
            self.test_data = data['data']
            self.length = min(5000, self.test_data.shape[0])

        # For generating data
        self.image_size = 64
        self.digit_size = 28

    def select_digit(self):
        while True:
            ind = random.randint(0, self.labels.shape[0] - 1)
            if self.selected_digits is None or len(self.selected_digits) == 0:
                return ind
            if self.labels[ind] in self.selected_digits:
                return ind

    def generate_moving_mnist(self):
        '''
        Get random trajectories for the digits and generate a video.
        '''
         
        sample = np.zeros((self.num_frames, self.image_size, self.image_size), dtype=np.float32)
        canvas_size = self.image_size - self.digit_size
        
        mnist_images = []
        for i in range(self.num_digits):
            ind = self.select_digit()
            mnist_images.append(Image.fromarray(self.images[ind]))
        
        # Randomly generate velocity (direction + speed), 
        if self.direcs is None:
            direcs = np.pi * (np.random.rand(self.num_digits) * 2 - 1)
        else:
            direcs = np.pi * np.full(self.num_digits, self.direcs)

        if self.speeds is None:
            speeds = np.random.randint(5, size=self.num_digits) + 2    
        else:
            speeds = np.full(self.num_digits, self.speeds)
            
        speeds = np.random.randint(5, size=self.num_digits) + 2
        veloc = np.asarray(
            [(speed * math.cos(direc), speed * math.sin(direc)) for direc, speed in zip(direcs, speeds)])
            
        # Generate initial positions for num_digits as tuples (x,y)
        positions = np.asarray(
            [(np.random.rand() * canvas_size, np.random.rand() * canvas_size) for _ in range(self.num_digits)])

        # Generate initial rotation angle for num_digits
        rotates = np.random.randint(0, 360, size=self.num_digits)
        rotate_increments = np.random.randint(5, 10, size=self.num_digits)

        # Generate new frames for the entire seq_length
        for frame_idx in range(self.num_frames): 

            canvas = np.zeros((self.image_size, self.image_size), dtype=np.float32)

            for i in range(self.num_digits):
                
                mnist_copy = Image.new('L', (self.image_size, self.image_size))
                cur_image = mnist_images[i]

                # first `num_rotas` digits do rotation
                if i < self.num_rotates:
                    cur_image = cur_image.rotate(rotates[i].astype(int))

                mnist_copy.paste(cur_image, tuple(positions[i].astype(int)))
                mnist_copy = np.array(mnist_copy) 
                canvas[mnist_copy > 0] = mnist_copy[mnist_copy > 0]
                
            next_pos = positions + veloc
            rotates = (rotates + rotate_increments) % 360

            # Iterate over velocity and see if we hit the wall. If yes, change direction
            if self.bounce:
                for i, pos in enumerate(next_pos):
                    for j, coord in enumerate(pos):
                        if coord < -2 or coord > canvas_size + 2:
                            veloc[i] = list(list(veloc[i][:j]) + [-1 * veloc[i][j]] + list(veloc[i][j + 1:]))

            # Make the permanent change to position by adding updated velocity
            # only undate position for the second digit
            for i in range(self.num_translates):
                positions[self.num_digits - i - 1] = positions[self.num_digits - i - 1] + veloc[self.num_digits - i - 1]

            # Add the canvas to the dataset array
            sample[frame_idx, ...] = canvas.clip(0, 255).astype(np.uint8)

        sample = sample[:, np.newaxis, ...]
        return sample

    def __getitem__(self, idx):

        if self.is_train:
            while True:
                sample = self.generate_moving_mnist()
                if self.bounce:
                    break
                if not self.bounce and np.sum(sample[-1] > 0) >= 128:
                    break
        else:
            sample = self.test_data[idx]

        input_seq = sample[:self.seq_len]
        output_seq = sample[self.seq_len: self.num_frames]
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
    