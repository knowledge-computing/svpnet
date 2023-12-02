import os 
import json
import time
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils.train_options import parse_args
from factory import load_model, load_data


def seed_everything(seed: int):

    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)


def train(args):
    
    gpu = args.gpu_id if torch.cuda.is_available() else None
    print(f'Device - GPU:{args.gpu_id}')
    
    train_loader, val_loader, test_loader = load_data(args)
    model = load_model(args)
    

    tensorboard_logger = TensorBoardLogger(
        save_dir=args.log_path,
        name='tensorboard_log',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        dirpath=args.model_path,
        filename='{epoch}-{val_loss:.5f}',
    )
    
    earlystop_callback = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.00000001, 
        mode='min',
        patience=args.patience, 
        verbose=True, 
    )
    
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        gpus= [gpu],
        log_every_n_steps=10,
        progress_bar_refresh_rate=0.5,
        logger=[tensorboard_logger],
        callbacks=[checkpoint_callback],
        check_val_every_n_epoch=1,
    )
    
    trainer.fit(model, train_loader, val_loader)

    # result_val = trainer.test(dataloaders=test_loader, ckpt_path='best')
    # args.test_loss = result_val[0]['test_loss']
    args.best_model_path = checkpoint_callback.best_model_path

    with open(args.output_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
def main():
    
    args = parse_args()
    train(args)
    
    
if __name__ == "__main__":
    main()
