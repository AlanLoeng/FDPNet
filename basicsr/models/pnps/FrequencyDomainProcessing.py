# ------------------------------------------------------------------------
# Copyright (c) 2025 Bolun Liang(https://github.com/AlanLoeng) All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from NAFNet (https://github.com/megvii-research/NAFNet)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.pnps import DeformableConv, CascadeDilatedConv

def initialize_weights_orthogonal(module):
    if isinstance(module, nn.Conv2d):
        nn.init.orthogonal_(module.weight) 
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
        

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=1)
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv1(x)
        return x
    
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size=3
        padding = (5 * (3 - 1)) // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, dilation=5, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x

class SimpleSpatialChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(SimpleSpatialChannelAttention, self).__init__()
        self.norm = LayerNorm2d(in_channels)
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.norm(x)
        channel_attn = self.channel_attention(x)
        spatial_attn = self.spatial_attention(x)
        return self.sigmoid(channel_attn * spatial_attn)

class AdaptiveLeakyReLU(nn.Module):
    def __init__(self, initial_negative_slope=0.01):
        super(AdaptiveLeakyReLU, self).__init__()
        self.negative_slope = nn.Parameter(torch.tensor(initial_negative_slope), requires_grad = True)

    def forward(self, x):
        return F.leaky_relu(x, negative_slope=self.negative_slope.item())

class FrequencyDomainProcessing(nn.Module):
    def __init__(self, in_channels, reduction=16, use_residual=True, refinement = True, attention = True):

        super(FrequencyDomainProcessing, self).__init__()
        assert refinement or attention is True

        self.use_residual = use_residual
        self.refinement=refinement
        self.attention = attention
        self.in_channels = in_channels
        self.norm1 = LayerNorm2d(in_channels)
        self.norm2 = LayerNorm2d(in_channels)        
        if attention:

            self.collectiveAttention = nn.Sequential(
                LayerNorm2d(in_channels*2),
                DepthwiseSeparableConv(in_channels*2,in_channels,3),
                SimpleSpatialChannelAttention(in_channels=in_channels),
                DepthwiseSeparableConv(in_channels,in_channels*2,3),
            )

            self.zeta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
            self.eta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

        if refinement == True:
            self.real_refinement = DepthwiseSeparableConv(in_channels,in_channels,3,padding=1)
            self.imaginary_refinement = DepthwiseSeparableConv(in_channels,in_channels,3,padding=1)
            
            self.collectiveRefinement = nn.Sequential(
                LayerNorm2d(in_channels),
                DepthwiseSeparableConv(in_channels,in_channels,3,padding=1)
            )
            self.Theta = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
            self.Iota = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1, 1)), requires_grad=True)
        self.delta = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)
        self.epsilon = nn.Parameter(torch.ones((1, in_channels, 1, 1)), requires_grad=True)


        self.apply(initialize_weights_orthogonal) 
        
    def forward(self, x):
        residual = x

        window_h = torch.hamming_window(x.shape[-2], periodic=False, dtype=x.dtype, layout=x.layout, device=x.device)
        window_w = torch.hamming_window(x.shape[-1], periodic=False, dtype=x.dtype, layout=x.layout, device=x.device)
        window = torch.outer(window_h, window_w)
        window = window.view(1, 1, x.shape[-2], x.shape[-1])
        x_windowed = x * window.expand_as(x) 

        x_freq = torch.fft.rfft2(x_windowed*self.epsilon+(1-self.epsilon)*x, dim=(-2, -1))
        real =self.norm1(torch.real(x_freq))
        imaginary = self.norm2(torch.imag(x_freq))    
        real = real * self.alpha
        imaginary = imaginary * self.beta

        if self.refinement == True:
            real = self.real_refinement(real)
            imaginary = self.imaginary_refinement(imaginary)

        # #collective attention
        if self.attention:
            x_cat = torch.cat((real,imaginary),dim=1)
            x_cat = self.collectiveAttention(x_cat)
            real_weights ,imaginary_weights = x_cat.chunk(2,dim=1)
            real = real * real_weights* self.zeta
            imaginary = imaginary * imaginary_weights*self.eta

        if self.refinement:
            mutiplied = real *imaginary
            mutiplied = self.collectiveRefinement(mutiplied)
            real = real + mutiplied* self.Theta
            imaginary = imaginary + mutiplied* self.Iota

        x_freq_refined = torch.complex(real, imaginary)
        x_spatial = torch.fft.irfft2(x_freq_refined, s=(x.shape[-2], x.shape[-1]), dim=(-2, -1))

        if self.use_residual:
            return self.delta* residual + x_spatial*self.gamma
        else:
            return x_spatial


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, 
                                 stride, padding, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))
    

