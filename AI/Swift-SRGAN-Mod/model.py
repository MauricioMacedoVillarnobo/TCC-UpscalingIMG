import torch
from torch import nn

# Depthwise convolution -> Pointwise convolution (Forms Depthwise-Separable Convolution)
class DepthwiseSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 1, bias = True):
        super().__init__()
        
        self.depthwise_layer = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_channels, bias = bias)
        self.pointwise_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1, bias = bias)
        
    def forward(self, x):
        return self.pointwise_layer(self.depthwise_layer(x))

# DepthwiseConvolution -> Batch Normalization -> PReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn = True, use_activation = True, generator = True, **kwargs):
        super().__init__()
        
        self.use_activation = use_activation
        self.sd_conv_layer = DepthwiseSepConv(in_channels, out_channels, **kwargs, bias = not use_bn)
        self.bn_layer = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation_layer = nn.PReLU(num_parameters = out_channels) if generator else nn.LeakyReLU(0.2, inplace = True)
        
    def forward(self, x):
        return self.activation_layer(self.bn_layer(self.sd_conv_layer(x))) if self.use_activation else self.bn_layer(self.sd_conv_layer(x))

# Depthwise-Separable Convolution -> Pixel Shuffle x2 -> PReLU
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super().__init__()
        
        self.sdconv_layer = DepthwiseSepConv(in_channels, in_channels * scale_factor**2, kernel_size = 3, stride = 1, padding = 1)
        self.ps_layer = nn.PixelShuffle(scale_factor) # in_channel * 4, H, W -> in_channel, H*2, W*2
        self.prelu_layer = nn.PReLU(num_parameters = in_channels)
    
    def forward(self, x):
        self.prelu_layer(self.ps_layer(self.sdconv_layer(x)))

# Covolution Block + Next Convolution Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.block1 = ConvBlock(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1)
        self.block2 = ConvBlock(in_channels, in_channels, kernel_size = 3, stride = 1, padding = 1, use_activation = False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x

# Generator    
class Generator(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = 64, num_blocks = 16, upscale_factor = 4):
        super().__init__()
        
        # Depthwise Convolution without Batch Normalization
        self.start = ConvBlock(in_channels, hidden_channels, use_bn = False, kernel_size = 9, stride = 1, padding = 4)
        
        # 16 Residual Blocks chain
        self.residual_blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        
        # Depthwise Convolution without PReLU
        self.midle_block = ConvBlock(hidden_channels, hidden_channels, kernel_size = 3, stride = 1, padding = 1, use_activation = False)
        
        # Upsample chain
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(hidden_channels, scale_factor = 2) for _ in range(upscale_factor // 2)])
        
        # Depthwise Convolution
        self.end = DepthwiseSepConv(num_blocks, in_channels, kernel_size = 9, stride = 1, padding = 4)
        
    def forward(self, x):
        initial = self.start(x)
        x = self.residual_blocks(initial)
        x = self.midle_block(x) + initial
        x = self.upsample_blocks(x)
        x = self.end(x)
        return torch.tanh(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, feature_map = (64, 64, 128, 128, 256, 256, 512, 512)):
        super().__init__()
        
        # Depthwise Convolution without Batch Normalization
        self.start = ConvBlock(in_channels, feature_map, kernel_size = 3, stride = 1, padding = 1, use_activation = True, use_bn = False, generator= False)
        
        # Discriminator Block Chain
        blocks = []
        for i, feature_map in enumerate(feature_map):
            blocks.append(ConvBlock(in_channels, feature_map, kernel_size = 3, stride = 1 + i % 2, padding = 1, use_activation = True, use_bn = False if i == 0 else True, generator= False))
            in_channels = feature_map
        self.blocks = nn.Sequential(*blocks)
        
        # Adaptative Average Pool (6x6) -> Linear (1024) -> Leaky ReLU -> Linear (1) -> Sigmoid
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten(), nn.Linear(516*6*6, 1024), nn.LeakyReLU(0.2, inplace = True), nn.Linear(1024, 1))
        
    def forward(self, x):
        x = self.start(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return torch.sigmoid(x) #Sigmoid