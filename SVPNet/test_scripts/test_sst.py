import os 
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as compare_ssim


def main(model, device, args):
    
    from data.sst import load_test_dataset
    test_dataset = load_test_dataset(args.data_path, args.seq_len, args.horizon)
    data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)
   
    num_samples = 0
    avg_mse = 0
    avg_mae = 0    
    avg_psnr = 0
    avg_ssim = 0
    frame_mse = [0] * args.horizon
    frame_mae = [0] * args.horizon    
    frame_psnr = [0] * args.horizon
    frame_ssim = [0] * args.horizon
 
    ground_truth, prediction = [], []  # these are for visualization

    model.eval()    
    with torch.no_grad():
        
        for it, data in enumerate(data_loader):
            
            x, y = data[0], data[1]
            batch_size = x.shape[0]
            
            frames = torch.cat([x, y], dim=1).to(device)
            masks = torch.FloatTensor(batch_size, args.horizon - 1, frames.size(2), 
                                      frames.size(3), frames.size(4)).fill_(0.0).to(device)
            if 'simvp' in args.model or 'tau' in args.model:
                out = model(x.to(device), masks)
            else:
                out = model(frames, masks)

                            
            frames = frames.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            
            # normalize back to original space
            norm_frames, norm_out = [], []
            for i in range(batch_size):
                _, day, region = test_dataset.samples[num_samples + i]
                max_v, min_v = test_dataset.max_value[0, 0, 0], test_dataset.min_value[0, 0, 0]
                z_mean, z_std = test_dataset.zone_mean[0, 0, region], test_dataset.zone_std[0, 0, region]
                d_mean, d_std = test_dataset.daily_zone_mean[0, day, region], test_dataset.daily_zone_std[0, day, region]
                
                tmp_frame = np.add(np.multiply(frames[i], max_v - min_v), min_v)
                tmp_out = np.add(np.multiply(out[i], max_v - min_v), min_v)
                    
                tmp_frame = np.add(np.multiply(tmp_frame, z_std), z_mean)
                tmp_frame = np.add(np.multiply(tmp_frame, d_std), d_mean)
                
                tmp_out = np.add(np.multiply(tmp_out, z_std), z_mean)
                tmp_out = np.add(np.multiply(tmp_out, d_std), d_mean)                

                norm_frames.append(tmp_frame)
                norm_out.append(tmp_out)
                    
            frames = np.stack(norm_frames)
            out = np.stack(norm_out)            
            
            if num_samples < 100:
                ground_truth.append(frames)
                prediction.append(out)
            
            frames = frames[:, -args.horizon:, ...]     
            out = out[:, -args.horizon:, ...]
            
            for i in range(args.horizon):
                
                frame_i = frames[:, i, ...]
                out_i = out[:, i, ...]
                mse = np.square(frame_i - out_i).sum()
                mae = np.abs(frame_i - out_i).sum()

                psnr = 0
                for b in range(batch_size):
                    mse_tmp = np.square(frame_i[b] - out_i[b]).mean()
                    psnr += 10 * np.log10(1 / mse_tmp)
                
                ssim = 0
                for b in range(batch_size):
                    tmp_a = frame_i[b, 0] * 255.0 / (frame_i[b, 0]).max()
                    tmp_b = out_i[b, 0] * 255.0 / (out_i[b, 0]).max()
                    ssim += compare_ssim((tmp_a).astype(np.uint8), 
                                         (tmp_b).astype(np.uint8))
                
                frame_mse[i] += mse
                frame_mae[i] += mae                
                frame_psnr[i] += psnr
                frame_ssim[i] += ssim
                avg_mse += mse
                avg_mae += mae 
                avg_psnr += psnr
                avg_ssim += ssim
                
            num_samples += batch_size
                                
    ground_truth = np.concatenate(ground_truth)
    prediction = np.concatenate(prediction)
            
    avg_mse = avg_mse / (num_samples * args.horizon)
    avg_mae = avg_mae / (num_samples * args.horizon)    
    avg_psnr = avg_psnr / (num_samples * args.horizon)
    avg_ssim = avg_ssim / (num_samples * args.horizon)
    
    print('mse: {:.4f}, mae: {:.4f}, psnr: {:.4f}, ssim: {:.4f}'.format(avg_mse, avg_mae, avg_psnr, avg_ssim))
    for i in range(args.horizon):
        print('frame {} - mse: {:.4f}, mae: {:.4f}, psnr: {:.4f}, ssim: {:.4f}'.format(i + args.seq_len + 1, 
                                                                                       frame_mse[i] / num_samples, 
                                                                                       frame_mae[i] / num_samples,
                                                                                       frame_psnr[i] / num_samples, 
                                                                                       frame_ssim[i] / num_samples))
        
    np.savez_compressed(
        os.path.join('{}/prediction.npz'.format(args.result_path)),
        ground_truth=ground_truth,
        prediction=prediction)


if __name__ == "__main__":
    main()
