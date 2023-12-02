import os 
import numpy as np
import torch
from utils.test_options import parse_args
from test_scripts import test_asii, test_sst, test_mnist, test_taxibj, test_turbulent
from factory import load_model


def main():
    
    args = parse_args() 
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    
    model = load_model(args).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device)['state_dict'], strict=False)
    print(f"Load model from {args.model}")
    
    if args.data_source.startswith('moving_mnist'):
        test_mnist.main(model, device, args)
    elif args.data_source.startswith('sst'):
        test_sst.main(model, device, args)
    elif args.data_source.startswith('asii'):
        test_asii.main(model, device, args)
    elif args.data_source.startswith('taxibj'):
        test_taxibj.main(model, device, args)
    elif args.data_source.startswith('turbulent'):
            test_turbulent.main(model, device, args)        

if __name__ == "__main__":
    
    import time
    start_time = time.time()
    main()
    print('Total Time: ', time.time() - start_time)
