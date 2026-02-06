import sys

sys.path.append("./src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.compress import *
from modules.bn import BatchNorm2d_
from modules.checkpointing import memory_optimization


class SeqToANNContainer(layer.SeqToANNContainer):
    """Stateless layer container that supports temporal chunking"""

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, xc):
        return (self.forward(xc),)


class NeuronConv2dBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        self.neuron = get_neuron(neuron_type, **kwargs)
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            BatchNorm2d_(out_channels),
        )
        self.x_compressor = "NullSpikeCompressor"

    def forward(self, x_seq):
        return self.conv(self.neuron(x_seq))

    def __spatial_split__(self):
        self.conv.x_compressor = "BitSpikeCompressor"
        return self.neuron, self.conv


class NeuronConv2dConv2dBN(nn.Module):
    def __init__(
        self,
        in_channels1,
        in_channels2,
        out_channels1,
        out_channels2,
        kernel_size1,
        kernel_size2,
        stride1=1,
        stride2=1,
        padding1=0,
        padding2=0,
        groups1=1,
        groups2=1,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        self.neuron = get_neuron(neuron_type, **kwargs)
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels1,
                out_channels1,
                kernel_size=kernel_size1,
                stride=stride1,
                padding=padding1,
                groups=groups1,
                bias=False,
            ),
            nn.Conv2d(
                in_channels2,
                out_channels2,
                kernel_size=kernel_size2,
                stride=stride2,
                padding=padding2,
                groups=groups2,
                bias=False,
            ),
            BatchNorm2d_(out_channels2),
        )
        self.x_compressor = "NullSpikeCompressor"

    def forward(self, x_seq):
        return self.conv(self.neuron(x_seq))

    def __spatial_split__(self):
        self.conv.x_compressor = "BitSpikeCompressor"
        return self.neuron, self.conv


class BNAndPadLayer(nn.Module):
    def __init__(
        self,
        pad_pixels,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super().__init__()
        self.bn = BatchNorm2d_(num_features, eps, momentum, affine, track_running_stats)
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                    self.bn.bias.detach()
                    - self.bn.running_mean
                    * self.bn.weight.detach()
                    / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0 : self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels :, :] = pad_values
            output[:, :, :, 0 : self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels :] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            BatchNorm2d_(out_channel),
        )
        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)


class RepConvBNNeuron(nn.Module):
    def __init__(self, in_channel, out_channel, neuron_type, **kwargs):
        super().__init__()
        self.conv = SeqToANNContainer(
            RepConv(in_channel, out_channel),
            BatchNorm2d_(out_channel),
        )
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x_seq):
        return self.neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.neuron


class SepConv(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=2,
        kernel_size=7,
        padding=3,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif_conv_bn1 = NeuronConv2dBN(
            dim,
            med_channels,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs,
        )
        self.lif_conv_bn2 = NeuronConv2dConv2dBN(
            in_channels1=med_channels,
            out_channels1=med_channels,
            kernel_size1=kernel_size,
            padding1=padding,
            groups1=med_channels,
            in_channels2=med_channels,
            out_channels2=dim,
            kernel_size2=1,
            stride2=1,
            neuron_type=neuron_type,
            **kwargs,
        )

    def forward(self, x):
        x = self.lif_conv_bn1(x)
        x = self.lif_conv_bn2(x)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, neuron_type="SJLIF", **kwargs):
        super().__init__()
        self.sep_conv = SepConv(dim=dim, neuron_type=neuron_type, **kwargs)
        self.lif_conv_bn1 = NeuronConv2dBN(
            dim,
            dim * mlp_ratio,
            kernel_size=3,
            padding=1,
            groups=1,
            bias=False,
            neuron_type=neuron_type,
            **kwargs,
        )
        self.lif_conv_bn2 = NeuronConv2dBN(
            dim * mlp_ratio,
            dim,
            kernel_size=3,
            padding=1,
            groups=1,
            bias=False,
            neuron_type=neuron_type,
            **kwargs,
        )

    def forward(self, x):
        x = self.sep_conv(x) + x
        x_feat = x
        x = self.lif_conv_bn1(x)
        x = self.lif_conv_bn2(x)
        x = x_feat + x
        return x


class MS_MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.lif_conv_bn1 = NeuronConv2dBN(
            in_features,
            hidden_features,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs,
        )
        self.lif_conv_bn2 = NeuronConv2dBN(
            hidden_features,
            out_features,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs,
        )
        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        x = self.lif_conv_bn1(x)
        x = self.lif_conv_bn2(x)
        return x


class HA3DCore(nn.Module):
    def __init__(self, num_heads, dim, neuron_type, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.bn = SeqToANNContainer(BatchNorm2d_(dim))
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        T, B, C, H, W = q.shape
        N = T * H * W

        q = (
            q.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )
        k = (
            k.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )
        v = (
            v.permute(1, 0, 3, 4, 2)  # [B, T, H, W, C]
            .flatten(1, 3)  # [B, THW, C]
            .reshape(B, N, self.num_heads, C // self.num_heads)  # [B, THW, M, C/M]
            .permute(0, 2, 1, 3)  # [B, M, THW, C/M]
            .contiguous()
        )
        x = (2 * k - 1).transpose(-2, -1) @ v
        x = (2 * q - 1) @ x
        x = x / (2 * self.dim)

        x = (
            x.permute(0, 2, 1, 3)  # [B, THW, M, C/M]
            .reshape(B, N, C)  # [B, THW, C]
            .reshape(B, T, H, W, C)  # [B, T, H, W, C]
            .permute(1, 0, 4, 2, 3)  # [T, B, C, H, W]
            .contiguous()
        )
        return self.neuron(self.bn(x))


class MS_Attention_3D_RepConv(nn.Module):
    def __init__(self, dim, num_heads=8, neuron_type="SJLIF", **kwargs):
        super().__init__()
        assert dim % num_heads == 0, (
            f"dim {dim} should be divided by num_heads {num_heads}."
        )
        self.dim = dim
        self.num_heads = num_heads

        self.head_lif = get_neuron(neuron_type, **kwargs)
        self.q_conv_lif = RepConvBNNeuron(dim, dim, neuron_type, **kwargs)
        self.k_conv_lif = RepConvBNNeuron(dim, dim, neuron_type, **kwargs)
        self.v_conv_lif = RepConvBNNeuron(dim, dim, neuron_type, **kwargs)
        self.attn = HA3DCore(num_heads, dim, neuron_type, **kwargs)
        self.tail = SeqToANNContainer(
            RepConv(dim, dim),
            BatchNorm2d_(dim),
        )

    def forward(self, x):
        x = self.head_lif(x)
        q, k, v = self.q_conv_lif(x), self.k_conv_lif(x), self.v_conv_lif(x)
        qkv = torch.stack([q, k, v], dim=0)
        x = self.attn(qkv)
        x = self.tail(x)
        return x


class MS_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, neuron_type="SJLIF", **kwargs):
        super().__init__()

        self.attn = MS_Attention_3D_RepConv(
            dim, num_heads=num_heads, neuron_type=neuron_type, **kwargs
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            neuron_type=neuron_type,
            **kwargs,
        )

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
        self,
        in_channels=2,
        embed_dims=256,
        kernel_size=3,
        stride=2,
        padding=1,
        first_layer=True,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        if first_layer:
            self.proj = NeuronConv2dBN(
                in_channels,
                embed_dims,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                neuron_type=neuron_type,
                **kwargs,
            )
        else:
            self.proj = SeqToANNContainer(
                nn.Conv2d(
                    in_channels,
                    embed_dims,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
                BatchNorm2d_(embed_dims),
            )

    def forward(self, x):
        return self.proj(x)


class Spiking_vit_MetaFormer(nn.Module):
    def __init__(
        self,
        in_channels=2,
        num_classes=11,
        embed_dim=[64, 128, 256],
        num_heads=[1, 2, 4],
        mlp_ratios=[4, 4, 4],
        depths=[6, 8, 6],
        T=32,
        neuron_type="SJLIF",
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
            neuron_type=neuron_type,
            **kwargs,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0] // 2,
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
            ]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_type=neuron_type,
            **kwargs,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[0],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
            ]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_type=neuron_type,
            **kwargs,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
            ]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [
                MS_ConvBlock(
                    dim=embed_dim[1],
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
            ]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
            neuron_type=neuron_type,
            **kwargs,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
                for j in range(6)
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
            neuron_type=neuron_type,
            **kwargs,
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    neuron_type=neuron_type,
                    **kwargs,
                )
                for j in range(2)
            ]
        )

        self.final_lif = get_neuron(neuron_type, **kwargs)
        self.head = (
            nn.Linear(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)

        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)

        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        return x  # T,B,C,N

    def forward(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = x.flatten(3).mean(3)
        x_lif = self.final_lif(x)
        x = self.head(x_lif.mean(0))
        return x


def GCMetaSpikformer(neuron_type, compress_x, level, **kwargs):
    net = Spiking_vit_MetaFormer(
        in_channels=3,
        num_classes=1000,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        depths=8,
        neuron_type=neuron_type,
        **kwargs,
    )
    return memory_optimization(
        net,
        (
            NeuronConv2dBN,
            NeuronConv2dConv2dBN,
            RepConvBNNeuron,
            HA3DCore,
            SeqToANNContainer,
        ),
        dummy_input=torch.zeros(4, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )
