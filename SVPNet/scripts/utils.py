import torch
import torch.nn as nn
import numpy as np


def activation_factory(name):
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(0.2, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'tanh':
        return nn.Tanh()
    if name is None or name == "identity":
        return nn.Identity()
    raise ValueError(f'Activation function `{name}` not yet implemented')


def init_net(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # Define the initialization function
        classname = m.__class__.__name__
        if classname in ('Conv2d', 'ConvTranspose2d', 'Linear'):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname == 'BatchNorm2d':
            if m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # Iterate the initialization function on the modules
    
    
# def reshape_patch(image, patch_size):
#     bs = image.size(0)
#     nc = image.size(1)
#     height = image.size(2)
#     weight = image.size(3)
#     x = image.reshape(bs, nc, int(height / patch_size), patch_size, int(weight / patch_size), patch_size)
#     x = x.transpose(2, 5)
#     x = x.transpose(4, 5)
#     x = x.reshape(bs, nc * patch_size * patch_size, int(height / patch_size), int(weight / patch_size))

#     return x


# def reshape_patch_back(image, patch_size):
#     bs = image.size(0)
#     nc = int(image.size(1) / (patch_size * patch_size))
#     height = image.size(2)
#     weight = image.size(3)
#     x = image.reshape(bs, nc, patch_size, patch_size, height, weight)
#     x = x.transpose(4, 5)
#     x = x.transpose(2, 5)
#     x = x.reshape(bs, nc, height * patch_size, weight * patch_size)

#     return x


class PositionalEncoding2d(nn.Module):
    def __init__(self, channels):
        
        super(PositionalEncoding2d, self).__init__()
        channels = int(np.ceil(channels / 4)*2)
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (B, C, H, W)
        :return: Positional Encoding Matrix of size (B, C, H, W)
        """
        
        B, C, H, W = tensor.shape
        
        pos_x = torch.arange(H, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(W, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((H, W, self.channels*2),device=tensor.device).type(tensor.type())
        emb[:, :, :self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :C].repeat(B, 1, 1, 1).permute(0, 3, 1, 2)
    
    
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding3d(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3d, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, ch, x, y, z)
        :return: Positional Encoding Matrix of size (batch_size, ch, x, y, z)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        B, C, x, y, z = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb(sin_inp_y).unsqueeze(1)
        emb_z = get_emb(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :C].repeat(B, 1, 1, 1, 1).permute(0, 4, 1, 2, 3)
