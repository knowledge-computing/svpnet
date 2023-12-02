import sys
import time
import argparse
import os
import glob
import json


def parse_args():
    
    # basic parameters
    parser = argparse.ArgumentParser(description='spatiotemporal-predictive-learning')
    
    # load data from file
    parser.add_argument('--data_source', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--seq_len', type=int, required=True)
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--result_root', type=str, default='./results')
    
    # training parameters
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--skip', action='store_true')
    
    parser.add_argument('--num_epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--loss_func', type=str, default='mse', help='mse | ce | L1 | smoothL1')
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--tau', type=int, default=3)
    
    parser.add_argument('--scheduler', type=str, default='ReduceLROnPlateau', help='MultiStepLR | ReduceLROnPlateau')
    parser.add_argument('--scheduler_patience', type=int, default=20)
    parser.add_argument('--scheduler_milestones', type=int, nargs='+', default=[300, 400])
    parser.add_argument('--scheduler_decay', type=float, default=0.1)
   
    args = parser.parse_args()
    args = data_configuration(args)
    args = output_configuration(args)
    
    for k, v in vars(args).items():
        print(f'{k} - {v}')
        
    with open(args.output_json_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    return args
    
    
def data_configuration(args):
    
    args.data_root = '/home/yaoyi/lin00786/data/spatiotemporal/data'
    
    if args.data_source.startswith('moving_mnist'):
        args.data_path = os.path.join(args.data_root, 'moving_mnist')
        args.vars = ['value']
        args.image_height = 64
        args.image_width = 64
        args.in_dim = 1
        args.h_dim = 64
        args.out_dim = 1  
        args.tau = 3
        
    elif args.data_source == 'taxibj':
        args.data_path = os.path.join(args.data_root, 'TaxiBJ')
        args.vars = ['inflow', 'outflow']
        args.image_height = 32
        args.image_width = 32      
        args.in_dim = 2
        args.h_dim = 128
        args.out_dim = 2 
        # args.tau = 3
        
    elif args.data_source == 'sst':
        args.data_path = os.path.join(args.data_root, 'SST')
        args.vars = ['value']
        args.image_height = 64
        args.image_width = 64        
        args.in_dim = 1
        args.h_dim = 128
        args.out_dim = 1  
        args.tau = 2
        
    elif args.data_source == 'turbulent':
        args.data_path = os.path.join(args.data_root, 'Turbulent')
        args.vars = ['u', 'v']
        args.image_height = 64
        args.image_width = 64        
        args.in_dim = 2
        args.h_dim = 128
        args.out_dim = 2  
        args.tau = 3
        
    else:
        raise NotImplementedError('Data source does not exist!')
    
    if not os.path.exists(args.data_path):
        print('Data path {} does not exist.'.format(args.data_path))
        sys.exit(-1)

    return args


def output_configuration(args):
    
    model_name = '{}_{}_seq{}_hoz{}'.format(args.model_name,
                                            args.data_source,
                                            args.seq_len,
                                            args.horizon)

    args.result_path = os.path.join(args.result_root, 
                                    model_name + '_' + str(int(time.time())))     
                
    args.model_path = os.path.join(args.result_path, 'models')
    args.log_path = os.path.join(args.result_path, 'logs')
    args.output_json_path = os.path.join(args.result_path, 'output.json')
  
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        os.makedirs(args.model_path)
        os.makedirs(args.log_path)
        
    return args
