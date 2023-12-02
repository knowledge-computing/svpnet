from torch.utils.data import DataLoader

def load_data(args):
    if args.data_source == 'moving_mnist_ori':
        from data.moving_mnist_live import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_source, 
                                                                args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)
    elif args.data_source.startswith('moving_mnist'):
        from data.moving_mnist_adv_live import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_source, 
                                                                args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)
    elif args.data_source == 'sst':
        from data.sst import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)
    elif args.data_source == 'asii':
        from data.asii import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)
    elif args.data_source == 'taxibj':
        from data.taxibj import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)
    elif args.data_source == 'turbulent':
        from data.turbulent import load_dataset
        train_dataset, val_dataset, test_dataset = load_dataset(args.data_path, 
                                                                args.seq_len,
                                                                args.horizon)        
    else:
        ValueError(f'Data loader `{args.data_source}` not yet implemented')
        
    print('#Train={}, #Val={}, #Test={}'.format(len(train_dataset), len(val_dataset), len(test_dataset)))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True,
        shuffle=True,
        drop_last=True)
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True,
        shuffle=False,
        drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, 
        pin_memory=True,        
        shuffle=False,
        drop_last=True)
    
    return train_loader, val_loader, test_loader
    
    
def load_encoder_decoder(data_source, skip):
    
    if data_source.startswith('moving_mnist'):
        from scripts.base_conv import MMNISTEncoder, MMNISTDecoder
        encoder = MMNISTEncoder(skip=skip)
        decoder = MMNISTDecoder(skip=skip)
    elif data_source == 'sst':
        from scripts.base_conv import SSTEncoder, SSTDecoder        
        encoder = SSTEncoder(skip=skip)
        decoder = SSTDecoder(skip=skip)
    elif data_source == 'asii':
        from scripts.base_conv import ASIIEncoder, ASIIDecoder        
        encoder = ASIIEncoder(skip=skip)
        decoder = ASIIDecoder(skip=skip)     
    elif data_source == 'taxibj':
        from scripts.base_conv import TaxiBJ16Encoder, TaxiBJ16Decoder        
        encoder = TaxiBJ16Encoder(skip=skip)
        decoder = TaxiBJ16Decoder(skip=skip)        
    elif data_source == 'taxibj1':
        from scripts.base_conv import TaxiBJ16Encoder1, TaxiBJ16Decoder1       
        encoder = TaxiBJ16Encoder1(skip=skip)
        decoder = TaxiBJ16Decoder1(skip=skip)        
    elif data_source == 'turbulent':
        from scripts.base_conv import TurbulentEncoder, TurbulentDecoder        
        encoder = TurbulentEncoder(skip=skip)
        decoder = TurbulentDecoder(skip=skip)
    else:
        raise ValueError(f'Encoder/Decoder `{data_source}` not yet implemented')
        return None
    return encoder, decoder


def load_model(args):
    
    encoder, decoder = load_encoder_decoder(args.data_source, args.skip)
    
    if args.model_name == 'svpnet':
        from scripts import svpnet
        model = svpnet.Trainer(encoder, decoder, args)
    else:
        raise ValueError(f'Model `{args.model_name}` not yet implemented')
        
    return model



