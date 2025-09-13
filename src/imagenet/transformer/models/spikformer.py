import sys

sys.path.append("./src")

import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.compress import *
from modules.bn import BatchNorm2d_, BatchNorm1d_
from modules.checkpointing import memory_optimization


class SeqToANNContainer(layer.SeqToANNContainer):
    """Stateless layer container that supports temporal chunking"""

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, xc):
        return (self.forward(xc),)


class Conv1dBNNeuron(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        neuron_type="SJLIF",
        **kwargs
    ):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            BatchNorm1d_(out_channels),
        )
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x_seq):
        return self.neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.neuron


class Conv2dBNNeuron(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        neuron_type="SJLIF",
        **kwargs
    ):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            ),
            BatchNorm2d_(out_channels),
        )
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x_seq):
        return self.neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.neuron


class NeuronMaxPool(nn.Module):

    def __init__(self, neuron_type, **kwargs):
        super().__init__()
        self.neuron = get_neuron(neuron_type, **kwargs)
        self.pool = SeqToANNContainer(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self._is_psn = neuron_type.endswith("PSN")

    def forward(self, x_seq):
        return self.pool(self.neuron(x_seq))

    def __tc_init_states__(self, x_seq):
        return [torch.zeros([], device=x_seq.device, dtype=x_seq.dtype)]

    def __tc_forward__(self, xc, v):
        sc, v = self.neuron.__tc_forward__(xc, v)
        yc = self.pool(sc)
        return yc, v


class Conv2dBNNeuronMaxPool(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        neuron_type, **kwargs
    ):
        super().__init__()
        self.conv_bn = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            BatchNorm2d_(out_channels),
        )
        self.neuron_pool = NeuronMaxPool(neuron_type, **kwargs)

    def forward(self, x):
        x = self.conv_bn(x)
        return self.neuron_pool(x)

    def __spatial_split__(self):
        return self.conv_bn, self.neuron_pool


class SSACore(nn.Module):

    def __init__(self, scale: float, neuron_type, **kwargs):
        super().__init__()
        self.scale = scale
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, qkv):
        # qkv.shape = [3, T, B, num_heads, num_patches, C//num_heads]
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]  # [T, B, num_heads, num_patches, C//num_heads]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * self.scale
        x = x.transpose(-1, -2)  # [T, B, num_heads, C//num_heads, num_patches]
        x = x.reshape(x.shape[0], x.shape[1], -1, x.shape[-1])
        return self.neuron(x)


class MLP(nn.Module):

    def __init__(
        self,
        neuron_type,
        in_features,
        hidden_features=None,
        out_features=None,
        **kwargs
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.conv1 = Conv2dBNNeuron(
            in_features,
            hidden_features,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.conv2 = Conv2dBNNeuron(
            hidden_features,
            out_features,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.conv1.x_compressor = "Uint8SpikeCompressor"

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SSA(nn.Module):

    def __init__(self, neuron_type, dim, num_heads=8, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} should be divided by num_heads {num_heads}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        self.qkv_network = Conv1dBNNeuron(
            dim,
            dim * 3,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.qkv_network.x_compressor = "Uint8SpikeCompressor"
        self.attn_network = SSACore(self.scale, neuron_type, **kwargs)
        self.proj_network = Conv1dBNNeuron(
            dim,
            dim,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs
        )

    def forward(self, x):
        # x.shape = [T, B, C, H, W]
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(3)
        T, B, C, N = x.shape

        qkv_conv_out = self.qkv_network(x)  # [T, B, 3C, N]
        qkv = qkv_conv_out.transpose(-1, -2).reshape(
            T, B, N, 3, self.num_heads, C // self.num_heads
        ).permute(3, 0, 1, 4, 2, 5).contiguous()  # [3, T, B, h, N, C//h]
        x = self.attn_network(qkv)  # [T, B, C, N]
        x = self.proj_network(x)  # [T, B, C, N]
        return x.reshape(T, B, C, H, W).contiguous()  # [T, B, C, H, W]


class Block(nn.Module):

    def __init__(self, neuron_type, dim, num_heads, mlp_ratio=4., **kwargs):
        super().__init__()

        self.attn = SSA(neuron_type, dim, num_heads, **kwargs)
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            neuron_type,
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            out_features=dim,
            **kwargs
        )

    def forward(self, x):  # not binary
        x = x + self.attn(x)  # not binary
        x = x + self.mlp(x)
        return x  # not binary!


class SPS(nn.Module):

    def __init__(
        self,
        neuron_type,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        in_channels=2,
        embed_dims=256,
        **kwargs
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = ((patch_size, patch_size)
                      if isinstance(patch_size, int) else patch_size)
        if len(patch_size) != 2:
            raise ValueError(
                f"patch_size should be a tuple of length 2 or an int, "
                f"but got {len(patch_size)}"
            )
        self.patch_size = patch_size
        self.C = in_channels
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv_0 = Conv2dBNNeuronMaxPool(
            in_channels,
            embed_dims // 8,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_1 = Conv2dBNNeuronMaxPool(
            embed_dims // 8,
            embed_dims // 4,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_2 = Conv2dBNNeuronMaxPool(
            embed_dims // 4,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_3 = Conv2dBNNeuronMaxPool(
            embed_dims // 2,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.positional_encoding = Conv2dBNNeuron(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )  # conv as learnable positional encoding

        self.proj_conv_0.x_compressor = "NullSpikeCompressor"

    def forward(self, x):
        # x is a float tensor
        x = self.proj_conv_0(x)
        x = self.proj_conv_1(x)
        x = self.proj_conv_2(x)
        x = self.proj_conv_3(x)

        x_feat = x
        x = self.positional_encoding(x)
        x = x + x_feat
        return x  # non-binary int tensor


class Spikformer(nn.Module):

    def __init__(
        self,
        neuron_type,
        T=4,
        in_channels=3,
        img_size_h=224,
        img_size_w=224,
        patch_size=16,
        num_classes=1000,
        embed_dims=512,
        num_heads=8,
        mlp_ratios=4,
        depths=8,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        kwargs["T"] = T

        self.patch_embed = SPS(
            neuron_type,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            **kwargs
        )
        self.block = nn.ModuleList([
            Block(
                neuron_type,
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                **kwargs
            ) for _ in range(depths)
        ])

        # classification head
        if num_classes > 0:
            self.head = nn.Linear(embed_dims, num_classes)
        else:
            self.head = nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        for blk in self.block:
            x = blk(x)  # [T, B, C, H, W]
        return x.flatten(3).mean(3)  # [T, B, C]

    def forward(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        x = self.forward_features(x)  # [T, B, C]
        x = self.head(x.mean(0))
        return x  # [B, num_classes]


def GCSpikformer(neuron_type, compress_x, level, **kwargs):
    net = Spikformer(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1dBNNeuron, Conv2dBNNeuron, Conv2dBNNeuronMaxPool, SSACore),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )
