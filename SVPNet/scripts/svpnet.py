import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from scripts.base_trainer import BaseTrainer
from scripts.ConvLSTM import ConvLSTM
from scripts.constrain_moments import K2M
from scripts.utils import PositionalEncoding2d, PositionalEncoding3d

        
class UpdateCell(nn.Module):
    def __init__(self, h_dim, kernel_size):
        super(UpdateCell, self).__init__()
        self.h_dim = h_dim
        padding = kernel_size // 2, kernel_size // 2
        self.conv1 = nn.Conv2d(h_dim, h_dim, 1)
        self.conv2 = nn.Conv2d(h_dim, h_dim, 1)
        self.conv3 = nn.Conv2d(h_dim, h_dim, 1)
        self.conv4 = nn.Conv2d(h_dim, h_dim, 1)
        self.gn = nn.GroupNorm(16, h_dim * 2)
        self.conv_gates = nn.Sequential(
            nn.Conv2d(in_channels=h_dim * 3,
                      out_channels=h_dim * 2,
                      kernel_size=kernel_size,
                      padding=padding),
            nn.GroupNorm(16, h_dim * 2))
        self.conv_can = nn.Sequential(
            nn.Conv2d(in_channels=h_dim * 3,
                      out_channels=h_dim,
                      kernel_size=kernel_size,
                      padding=padding),
            nn.GroupNorm(16, h_dim))
        self.out = nn.Sequential(
            nn.Conv2d(h_dim, h_dim, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(h_dim, h_dim, 1))
        
    def forward(self, x, prev_x, cur_predict, last_predict, cur_update, last_update, hidden):
        # delta_1 = self.conv1(x - prev_x)
        delta_2 = self.conv2(x - cur_predict)
        # delta_3 = self.conv3(post - prev_post)
        delta_4 = self.conv4(cur_update - last_predict)
        input_data = torch.cat([delta_2, delta_4], dim=1)
        input_data = self.gn(input_data)
        combined = torch.cat([input_data, hidden], dim=1)
        combined_conv = self.conv_gates(combined)
        gamma, beta = torch.split(combined_conv, self.h_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)
        combined = torch.cat([input_data, reset_gate * hidden], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)
        hidden = (1 - update_gate) * hidden + update_gate * cnm
        out = self.out(hidden)
        new_update = cur_predict + (x - cur_predict) * torch.sigmoid(out)
        return out, hidden


class CrossAttentionLayer(nn.Module):
    def __init__(self, h_dim, out_dim):    
        super(CrossAttentionLayer, self).__init__()
        self.num_heads = 8
        self.head_dim = h_dim // self.num_heads
        self.query_proj = nn.Conv2d(h_dim, self.num_heads * self.head_dim, 1)
        self.key_proj = nn.Conv3d(h_dim, self.num_heads * self.head_dim, 1)
        self.value_proj = nn.Conv3d(h_dim, self.num_heads * self.head_dim, 1)
        self.out_proj = nn.Conv2d(h_dim, out_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.query_proj.weight)
        torch.nn.init.xavier_uniform_(self.key_proj.weight)
        torch.nn.init.xavier_uniform_(self.value_proj.weight)
        torch.nn.init.constant_(self.query_proj.bias, 0.)
        torch.nn.init.constant_(self.key_proj.bias, 0.)  
        torch.nn.init.constant_(self.value_proj.bias, 0.)   

    def forward(self, query, key, value):
        # assert query.shape == key.shape == value.shape
        B, C, T, H, W = key.shape
        Q_h = self.query_proj(query)
        K_h = self.key_proj(key).reshape(B, C, T * H * W)
        V_h = self.value_proj(value).reshape(B, C, T * H * W)
        
        Q_h = Q_h.reshape(B, self.num_heads, self.head_dim, H * W).permute(0, 3, 1, 2)
        K_h = K_h.reshape(B, self.num_heads, self.head_dim, T * H * W).permute(0, 3, 1, 2)
        V_h = V_h.reshape(B, self.num_heads, self.head_dim, T * H * W).permute(0, 3, 1, 2)
        energy = torch.einsum("bpnd,bqnd->bpqn", [Q_h, K_h])
        A_h = torch.softmax(energy / ((self.num_heads * self.head_dim) ** 0.5), dim=-2)
        out = torch.einsum("bpqn,bqnd->bpnd", [A_h, V_h]).reshape(
                B, H * W, self.num_heads * self.head_dim).reshape(
                B, H, W, self.num_heads * self.head_dim).permute(0, 3, 1, 2)
        out = self.out_proj(out)
        return out, A_h

    
class PredictCell(nn.Module):
    def __init__(self, h_dim, phys_dim, phys_kernel_size):
        super(PredictCell, self).__init__()
        self.h_dim = h_dim
        self.F = nn.Conv2d(in_channels=h_dim, 
                           out_channels=phys_dim, 
                           kernel_size=phys_kernel_size, 
                           stride=1, 
                           padding=(phys_kernel_size // 2, phys_kernel_size // 2))
        self.gn = nn.GroupNorm(7, phys_dim)
    
        self.lin1 = nn.Sequential(
            nn.GroupNorm(8, h_dim),
            nn.Conv2d(h_dim, phys_dim, 1))
        
        self.lin2 = nn.Sequential(
            nn.GroupNorm(7, phys_dim),
            nn.Conv2d(phys_dim, h_dim, 1))
        
        self.pos_enc_2d = PositionalEncoding2d(h_dim)
        self.pos_enc_3d = PositionalEncoding3d(h_dim)
        self.cross_attn = CrossAttentionLayer(h_dim, h_dim)
        
    def forward(self, x, prev_x, new_update):         
        B, C, H, W = x.shape
        h_grad = self.gn(self.F(new_update))
        
        pos_enc_2d = self.pos_enc_2d(x)
        pos_enc_3d = self.pos_enc_3d(prev_x)
        
        q = new_update + pos_enc_2d
        k, v = prev_x + pos_enc_3d, prev_x 
        coef, A_ = self.cross_attn(q, k, v)
        coef = self.lin1(coef)
        
        new_predict = self.lin2(coef * h_grad) + new_update
        return new_predict
    
    
class PhysCell(nn.Module):
    def __init__(self, h_dim, h_kernel_size, phys_dim, phys_kernel_size):
        super(PhysCell, self).__init__()                  
        self.h_dim = h_dim
        self.phys_dim = phys_dim
        self.update_cell = UpdateCell(h_dim, h_kernel_size)
        self.predict_cell = PredictCell(h_dim, phys_dim, phys_kernel_size)
        
    def forward(self, cur_phys, last_phys, temporal_list, cur_predict, cur_update, last_predict, last_update, h_weight):
        new_update, h_weight = self.update_cell(cur_phys, last_phys, 
                                                cur_predict, last_predict, 
                                                cur_update, last_update, h_weight)
        temporal_list = torch.cat([temporal_list[:, :, 1:], new_update.unsqueeze(2)], dim=2)
        new_predict = self.predict_cell(cur_phys, temporal_list, new_update)
        return new_predict, new_update, h_weight


class Encoder2D(nn.Module):
    def __init__(self, in_c, out_c):
        super(Encoder2D, self).__init__()
        self.c1 = nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, 1, 1),
                nn.GroupNorm(16, out_c),
                nn.LeakyReLU(0.2, inplace=True))
        self.c2 = nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, 1, 1),
                nn.GroupNorm(16, out_c),
                nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        h1 = self.c1(x)  
        h2 = self.c2(h1)     
        return h2

    
class Decoder2D(nn.Module):
    def __init__(self, in_c, out_c):
        super(Decoder2D, self).__init__()
        self.upc1 = nn.Sequential(
            nn.ConvTranspose2d(in_c, in_c, 3, 1, 1),
            nn.GroupNorm(16, in_c),
            nn.LeakyReLU(0.2, inplace=True))
        self.upc2 = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 3, 1, 1),
            nn.GroupNorm(16, out_c),
            nn.LeakyReLU(0.2, inplace=True))
    def forward(self, x):
        d1 = self.upc1(x) 
        d2 = self.upc2(d1)  
        return d2       
    
    
class PhysNet(nn.Module):
    def __init__(
        self, 
        h_dim,
        h_kernel_size,        
        phys_dim,
        phys_kernel_size, 
        num_layers, 
        tau,
        device
    ):
        super(PhysNet, self).__init__()                  
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device
        self.tau = tau
        
        self.temporal_list = []
        self.last_phys = []
        self.cur_update = []
        self.last_update = []
        self.cur_predict = []
        self.last_predict = []
        self.h_weight = []
        
        phys_cells = []
        for i in range(num_layers):
            phys_cells.append(PhysCell(h_dim, h_kernel_size, phys_dim, phys_kernel_size))
        self.phys_cells = nn.ModuleList(phys_cells)
        
    def forward(self, x, first_timestep=False):
        B, _, H, W = x.shape        
        if first_timestep:
            self.init_hidden(batch_size=B, image_size=(H, W))
        
        for i in range(self.num_layers):
            if i == 0:
                in_phys = x
            else:
                in_phys = self.cur_predict[i-1]
                
            # self.temporal_list[i] = torch.cat([self.temporal_list[i][:, :, 1:], in_phys.unsqueeze(2)], dim=2)
            new_predict, new_update, self.h_weight[i] = self.phys_cells[i](in_phys, self.last_phys[i], self.temporal_list[i],
                                                                           self.cur_predict[i], self.last_predict[i], 
                                                                           self.cur_update[i], self.last_update[i], 
                                                                           self.h_weight[i])
            
            self.temporal_list[i] = torch.cat([self.temporal_list[i][:, :, 1:], new_update.unsqueeze(2)], dim=2)
            
            self.last_phys[i] = in_phys
            self.last_predict[i] = self.cur_predict[i]
            self.cur_predict[i] = new_predict            
            self.last_update[i] = self.cur_update[i]
            self.cur_update[i] = new_update

        return self.cur_predict, self.cur_predict
    
    def init_hidden(self, batch_size, image_size):
        H, W = image_size
        self.temporal_list = []
        self.last_phys = []
        self.cur_update = []
        self.last_update = []
        self.cur_predict = []
        self.last_predict = []
        self.h_weight = []
        for i in range(self.num_layers):
            self.temporal_list.append(torch.zeros(batch_size, self.h_dim, self.tau, H, W).to(self.device))
            self.last_phys.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))            
            self.cur_update.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))
            self.last_update.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))
            self.cur_predict.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))
            self.last_predict.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))
            self.h_weight.append(torch.zeros(batch_size, self.h_dim, H, W).to(self.device))

        
class Net(torch.nn.Module):
    def __init__(self, encoder, decoder, h_dim, h_kernel_size, phys_dim, phys_kernel_size, num_layers, tau, device):
        super(Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.physnet = PhysNet(h_dim, h_kernel_size, phys_dim, phys_kernel_size, 1, tau, device)
        self.resnet = ConvLSTM(h_dim, [h_dim] * num_layers, num_layers, h_kernel_size, device) 
        self.enc = Encoder2D(h_dim, h_dim)
        self.dec_phys = Decoder2D(h_dim, h_dim)
        self.dec_res = Decoder2D(h_dim, h_dim)
                
    def forward(self, x, first_timestep=False):
        in_x, skips = self.encoder(x)
        in_phys = self.enc(in_x)
        in_res = in_x - in_phys
        # input_conv = self.encoder_Er(en_x)     
        
        h_phys, _ =  self.physnet(in_phys, first_timestep)
        h_res, _ = self.resnet(in_res, first_timestep)
        h_phys = self.dec_phys(h_phys[-1])
        h_res = self.dec_res(h_res[-1])
        out = h_phys + h_res   
        out_phys = self.decoder(h_phys, skips=skips) 
        out_res = self.decoder(h_res, skips=skips)
        out = self.decoder(out, skips=skips)
        return out, out_phys, out_res
           
    
class Trainer(BaseTrainer):
    def __init__(self, encoder, decoder, args):
        super().__init__(args)
        self.constraints = torch.zeros((49, 7, 7)).to(self.gpu)
        self.alpha = args.alpha
        self.tau = args.tau
        
        ind = 0
        for i in range(0,7):
            for j in range(0,7):
                self.constraints[ind,i,j] = 1
                ind +=1 
                
        self.net = Net(encoder, 
                       decoder, 
                       h_dim=self.h_dim,
                       h_kernel_size=3, 
                       phys_dim=49, 
                       phys_kernel_size=7, 
                       num_layers=3, 
                       tau=self.tau,
                       device=self.gpu)

    def forward(self, frames, masks):
        next_frames = []            
        for t in range(self.seq_len + self.horizon - 1): 
            if t < self.seq_len:
                x = frames[:, t, :, :, :]
            else:
                mask = masks[:, t - self.seq_len]
                x = mask * frames[:, t, :, :, :] + (1 - mask) * next_frames[-1]
                # x = next_frames[-1]
                
            out_frame, out_phys, out_res = self.net(x, first_timestep=(t==0))    
            next_frames.append(out_frame)

        next_frames = torch.stack(next_frames, dim=1) 
        return next_frames
    
    
    def compute_train_loss(self, y, y_hat):
        mse_loss = self.loss_fn['mse'](y, y_hat)
        moment_loss = 0
        k2m = K2M([7,7]).to(self.gpu)
        for i in range(len(self.net.physnet.phys_cells)):
            for b in range(self.net.physnet.h_dim):
                filters = self.net.physnet.phys_cells[i].predict_cell.F.weight[:,b,:,:] # (nb_filters,7,7)
                m = k2m(filters.double()) 
                m  = m.float()
                moment_loss += self.loss_fn['mse'](m, self.constraints)
        return mse_loss, moment_loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        frames = torch.cat([x, y], dim=1)
        frames, masks = self.set_train_inputs(frames)
        out = self(frames, masks)
        mse_loss, moment_loss = self.compute_train_loss(frames[:, 1:], out)
        self.log(f'train_loss', mse_loss)
        return mse_loss * self.alpha + moment_loss
    
   