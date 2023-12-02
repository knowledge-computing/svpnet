import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


class BaseTrainer(pl.LightningModule):
    def __init__(self, args):
        super(BaseTrainer, self).__init__()
        # self.save_hyperparameters()
        self.h_dim = args.h_dim
        self.num_layers = args.num_layers
        self.seq_len = args.seq_len
        self.horizon = args.horizon
        self.batch_size = args.batch_size
        self.image_height = args.image_height
        self.image_width = args.image_width        
        self.kernel_size = 3
        self.eta = 1.0
        self.skip = args.skip
        self.scheduler = args.scheduler
        self.scheduler_milestones = list(args.scheduler_milestones)
        self.scheduler_decay = args.scheduler_decay
        self.scheduler_patience = args.scheduler_patience
        self.lr = args.lr
        self.loss_func = args.loss_func
        self.loss_fn = {
            'smoothL1': nn.SmoothL1Loss(), 
            'L1': nn.L1Loss(), 
            'mse': nn.MSELoss(),
            'ce': nn.CrossEntropyLoss()}
        
        self.gpu = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else "cpu")
        
    def forward(self, frames, masks):
        pass
    
    def compute_train_loss(self, y, y_hat):
        pass

    def compute_test_loss(self, y, y_hat):
        return self.loss_fn['mse'](y, y_hat)
    
    def set_train_inputs(self, frames):
        B, T, C, H, W = frames.shape    
        self.eta = np.maximum(0 , 1 - self.current_epoch * 0.02) 
        random_flip = np.random.random_sample((B, self.horizon - 1))
        true_token = (random_flip < self.eta)
        one = torch.FloatTensor(1, 1, C, H, W).fill_(1.0).to(self.gpu)
        zero = torch.FloatTensor(1, 1, C, H, W).fill_(0.0).to(self.gpu)
        masks = []
        for b in range(B):
            masks_t = []
            for t in range(self.horizon - 1):
                if true_token[b, t]:
                    masks_t.append(one)
                else:
                    masks_t.append(zero)
            mask = torch.cat(masks_t, 1)
            masks.append(mask)
        masks = torch.cat(masks, 0)
        return frames, masks

    def set_test_inputs(self, frames):
        B, T, C, H, W = frames.shape
        masks = torch.FloatTensor(B, self.horizon - 1, C, H, W).fill_(0.0).to(self.gpu)
        return frames, masks

    def training_step(self, batch, batch_idx):
        pass
    
    def training_epoch_end(self, outputs):
        for param_group in self.optimizers().param_groups:
            print(param_group["lr"])  
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        frames = torch.cat([x, y], dim=1)
        frames, masks = self.set_test_inputs(frames)
        out = self(frames, masks)
        out = out[:, -self.horizon:, ...]
        loss = self.compute_test_loss(y, out)
        self.log(f'val_loss', loss)
        return loss
          
    def test_step(self, batch, batch_idx):
        x, y = batch
        frames = torch.cat([x, y], dim=1)
        frames, masks = self.set_test_inputs(frames)
        out = self(frames, masks)
        out = out[:, -self.horizon:, ...]
        loss = self.compute_test_loss(y, out)
        self.log(f'test_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.99))
        
        if self.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.scheduler_patience, factor=0.1, verbose=True)
        elif self.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.scheduler_milestones, gamma=self.scheduler_decay)
        else:
            raise NotImplementedError
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler":scheduler,
                "monitor": "val_loss"}}