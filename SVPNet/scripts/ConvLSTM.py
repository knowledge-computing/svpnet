import torch.nn as nn
import torch


class ConvLSTMCell(nn.Module):
    def __init__(self, in_dim, h_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.h_dim = h_dim
        padding = kernel_size // 2, kernel_size // 2
        self.conv = nn.Conv2d(in_channels=in_dim + h_dim,
                              out_channels=4 * h_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)
        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels + h_channels,
        #               out_channels=4 * h_channels,
        #               kernel_size=kernel_size,
        #               padding=padding,
        #               bias=True),
        #     nn.GroupNorm(4 * h_channels // 32, 4 * h_channels))
    
    def forward(self, input_data, prev_state):
        h_prev, c_prev = prev_state
        combined = torch.cat((input_data, h_prev), dim=1)  
        combined_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_output, self.h_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_cur = f * c_prev + i * g
        h_cur = o * torch.tanh(c_cur)
        return h_cur, c_cur

    
class ConvLSTM(nn.Module):
    def __init__(
        self, 
        in_dim, 
        h_dims, 
        num_layers,
        kernel_size, 
        device
    ):    
        super(ConvLSTM, self).__init__()  
        self.in_dim = in_dim
        self.h_dims = h_dims
        self.num_layers = num_layers
        layer_list = []
        for i in range(num_layers):
            cur_in_dim = in_dim if i == 0 else h_dims[i - 1]
            layer_list.append(ConvLSTMCell(in_dim=cur_in_dim,
                                           h_dim=h_dims[i],
                                           kernel_size=kernel_size))
        self.layer_list = nn.ModuleList(layer_list)
        self.H, self.C = [], []
        self.device = device
            
    def forward(self, x, first_timestep=False):
        if first_timestep:
            self.init_hidden(batch_size=x.size(0),
                             image_size=(x.size(2), x.size(3)))
        
        for i, layer in enumerate(self.layer_list):
            if i == 0:
                self.H[i], self.C[i] = layer(input_data=x, prev_state=[self.H[i], self.C[i]])
            else:
                self.H[i], self.C[i] = layer(input_data=self.H[i - 1], prev_state=[self.H[i], self.C[i]])

        return self.H, (self.H, self.C)
    
    def init_hidden(self, batch_size, image_size):
        self.H, self.C = [], []
        for i in range(self.num_layers):
            hidden_state = torch.zeros(batch_size, self.h_dims[i], image_size[0], image_size[1])
            cell_state = torch.zeros(batch_size, self.h_dims[i], image_size[0], image_size[1])
            self.H.append(hidden_state.to(self.device))
            self.C.append(cell_state.to(self.device))
            