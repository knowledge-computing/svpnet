import sys
import time
import argparse
import os
import glob
import json


def parse_args():
    
    parser = argparse.ArgumentParser(description='video_prediction')
    
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--data_source', type=str, default='moving_mnist_2_01')    
    parser.add_argument('--result_path', type=str, default='./results_new')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU id')
    parser.add_argument('--horizon', type=int, default=-1)

    args = parser.parse_args()
    
    args.result_path = os.path.join(args.result_path, args.model)            
    args.model_path = os.path.join(args.result_path, 'models')
    
    output_json_path = os.path.join(args.result_path, 'output.json')
    with open(output_json_path, "r") as f:
        output_json = json.load(f)
            
    for k, v in output_json.items():
        if k not in vars(args).keys():
            args.__dict__.update({k: v})
    
    if args.horizon == -1:
        args.horizon = output_json['horizon']
        
    if not os.path.exists(args.model_path):
        print('Model path {} does not exist.'.format(args.model_path))
        sys.exit(-1)
        
    model_path = '{}/*.ckpt'.format(args.model_path)
    model_path = glob.glob(model_path)
    assert len(model_path)==1, f'All these files were found: {model_path}'
    args.model_path= model_path[0]
            
    return args

