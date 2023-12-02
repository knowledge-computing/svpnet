import torch
import torch.nn as nn
from scripts.utils import activation_factory


def make_conv_block(conv, activation, norm):
    out_channels = conv.out_channels
    modules = [conv]
    if norm == 'bn':
        modules.append(nn.BatchNorm2d(out_channels))
    elif norm == 'gn':
        modules.append(nn.GroupNorm(16, out_channels))
    if activation is not None:
        modules.append(activation_factory(activation))
    return nn.Sequential(*modules)

    
class MMNISTEncoder(nn.Module):
    def __init__(self, skip=False):
        super(MMNISTEncoder, self).__init__()
        self.skip = skip
        self.h_dim = 64
        self.sr = 4
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(1, 16, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(16, 32, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None

        
class MMNISTDecoder(nn.Module):
    def __init__(self, skip=False):
        super(MMNISTDecoder, self).__init__()
        self.skip = skip
        self.h_dim = 64
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(64 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 2, 1, output_padding=1), 
                            activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 16, 3, 2, 1, output_padding=1),
                            activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(16 * coef, 1, 3, 1, 1), activation=None, norm=None)])        
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)    

    
class TaxiBJ16Encoder(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJ16Encoder, self).__init__()
        self.skip = skip
        self.h_dim = 128
        self.sr = 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(2, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 128, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None
        
        
class TaxiBJ16Decoder(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJ16Decoder, self).__init__()
        self.skip = skip
        self.h_dim = 128
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(128 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 2, 1, output_padding=1),
                            activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 2, 3, 1, 1), activation=None, norm=None)])
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)     

    
class TaxiBJEncoder(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJEncoder, self).__init__()
        self.skip = skip
        self.h_dim = 128
        self.sr = 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(2, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 128, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None
        
        
class TaxiBJDecoder(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJDecoder, self).__init__()
        self.skip = skip
        self.h_dim = 128
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(128 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 2, 1, output_padding=1),
                            activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 2, 3, 1, 1), activation=None, norm=None)])
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)     
    
    
class TaxiBJ16Encoder1(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJ16Encoder1, self).__init__()
        self.skip = skip
        self.h_dim = 128
        self.sr = 2
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(2, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 128, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None
        
        
class TaxiBJ16Decoder1(nn.Module):
    def __init__(self, skip=False):
        super(TaxiBJ16Decoder1, self).__init__()
        self.skip = skip
        self.h_dim = 128
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(128 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 2, 3, 1, 1), activation=None, norm=None)])
        
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)     

    
    
    
class TurbulentEncoder(nn.Module):
    def __init__(self, skip=False):
        super(TurbulentEncoder, self).__init__()
        self.skip = skip
        self.h_dim = 128 
        self.sr = 4
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(2, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 128, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None
        
        
class TurbulentDecoder(nn.Module):
    def __init__(self, skip=False):
        super(TurbulentDecoder, self).__init__()
        self.h_dim = 128 
        self.skip = skip
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(128 * coef, 128, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(128 * coef, 64, 3, 2, 1, output_padding=1), 
                            activation='leaky_relu', norm=None),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 2, 1, output_padding=1),
                            activation='leaky_relu', norm=None),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 2, 3, 1, 1), activation=None, norm=None)])
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)     


class SSTEncoder(nn.Module):
    def __init__(self, skip=False):
        super(SSTEncoder, self).__init__()
        self.skip = skip
        self.h_dim = 128 
        self.sr = 4
        self.conv = nn.ModuleList([
            make_conv_block(nn.Conv2d(1, 32, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(32, 64, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(64, 128, 3, 2, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.Conv2d(128, 128, 3, 1, 1), activation='leaky_relu', norm='gn')])
        
    def forward(self, x):
        skips = []
        out = x
        for layer in self.conv:
            out = layer(out)
            skips.append(out)
        if self.skip:
            return out, skips[::-1]
        else:
            return out, None
        
        
class SSTDecoder(nn.Module):
    def __init__(self, skip=False):
        super(SSTDecoder, self).__init__()
        self.h_dim = 128 
        self.skip = skip
        coef = 2 if skip else 1
        self.conv = nn.ModuleList([
            make_conv_block(nn.ConvTranspose2d(128 * coef, 128, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(128 * coef, 64, 3, 2, 1, output_padding=1), 
                            activation='leaky_relu', norm=None),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 64, 3, 1, 1), activation='leaky_relu', norm='gn'),
            make_conv_block(nn.ConvTranspose2d(64 * coef, 32, 3, 2, 1, output_padding=1),
                            activation='leaky_relu', norm=None),
            make_conv_block(nn.ConvTranspose2d(32 * coef, 1, 3, 1, 1), activation=None, norm=None)])
        self.last_activation = activation_factory('sigmoid')
        
    def forward(self, x, skips=None):
        assert skips is None and not self.skip or skips is not None and self.skip
        out = x
        for i, layer in enumerate(self.conv):
            if skips is not None:
                out = torch.cat([out, skips[i]], 1)
            out = layer(out)
        return self.last_activation(out)     
    
    