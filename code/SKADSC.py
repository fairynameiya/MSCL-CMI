from collections import OrderedDict
import torch
from torch import nn



class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        self.d = max(L, channel // reduction)
        self.convs = nn.ModuleList()
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList()
        for _ in kernels:
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, 0)  # k,bs,c,h,w

        U = sum(conv_outs)  # bs,c,h,w
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        weights = [fc(Z).view(bs, c, 1, 1) for fc in self.fcs]
        attention_weights = torch.stack(weights, 0)  # k,bs,c,1,1
        attention_weights = self.softmax(attention_weights)

        V = (attention_weights * feats).sum(0)  # bs,c,h,w
        return V



import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv2d(nn.Module):
    """2D 深度可分离卷积模块：Depthwise + Pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), dropout=0.1):
        super(DepthwiseSeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=(kernel_size[0]//2, kernel_size[1]//2),
            groups=in_channels  # Depthwise卷积
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.relu(x)      # 激活
        x = self.bn(x)        # 一次 BN
        x = self.dropout(x)
        return x



class ResidualBlock2d(nn.Module):
    def __init__(self, channels, kernel_size=(3,3), dropout=0.1):
        super().__init__()
        self.conv_block = DepthwiseSeparableConv2d(
            channels, channels, kernel_size, dropout
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual        # 直接相加
        return F.relu(out)     # 只有最后一次激活

class ResNetDWConv2d(nn.Module):
    """适用于短RNA/多通道特征的 2D ResNet + DWConv 模型（保留4D输出）"""
    def __init__(self, input_channels=5, num_blocks=3, filters=5, dropout=0.2, use_global_pool=False):
        super(ResNetDWConv2d, self).__init__()

        self.use_global_pool = use_global_pool  # 是否使用全局池化

        # 初始卷积层
        self.conv_in = nn.Conv2d(input_channels, filters, kernel_size=(3,3), padding=1)
        self.bn_in = nn.BatchNorm2d(filters)

        # 堆叠多个残差块
        self.res_blocks = nn.Sequential(
            *[ResidualBlock2d(filters, kernel_size=(3,3), dropout=dropout) for _ in range(num_blocks)]
        )



    def forward(self, x):
        """
        x: [batch, channels, fm, seq_len]
        """
        # 初始卷积
        x = self.conv_in(x)
        # print("卷积", x.shape)
        x = self.bn_in(x)
        # print("BN", x.shape)
        x = F.relu(x)
        # print("RELU", x.shape)
        # 残差块
        x = self.res_blocks(x)
        # print("残差链接",x.shape)


        return x


