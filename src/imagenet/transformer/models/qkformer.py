import sys

sys.path.append("./src")

import torch.nn as nn

from modules.neuron import get_neuron
from modules.compress import *
from modules.bn import BatchNorm2d_
from modules.checkpointing import memory_optimization

from .spikformer import Conv1dBNNeuron, Conv2dBNNeuron
from .spikformer import MLP, SSA, SSACore, SeqToANNContainer


class QKACore(nn.Module):

    def __init__(self, neuron_type, **kwargs):
        super().__init__()
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, qk):
        # qk.shape = [T, B, 2, num_heads, C//num_heads, num_patches]
        q, k = qk[:, :, 0], qk[:, :, 1]
        q = torch.sum(q, dim=3, keepdim=True)
        q = self.neuron(q)  # [T, B, num_heads, 1, num_patches]; token-wise
        k = torch.mul(q, k)  # [T, B, num_heads, C//num_heads, num_patches]
        return k.flatten(2, 3)  # [T, B, C, num_patches]


class QKA(nn.Module):

    def __init__(self, neuron_type, dim, num_heads=8, **kwargs):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} should be divided by num_heads {num_heads}."
            )
        self.dim = dim
        self.num_heads = num_heads

        self.qk_network = Conv1dBNNeuron(
            dim,
            dim * 2,
            kernel_size=1,
            stride=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.qk_network.x_compressor = "Uint8SpikeCompressor"
        self.attn_network = QKACore(neuron_type, **kwargs)
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

        qk_conv_out = self.qk_network(x)
        qk = qk_conv_out.reshape(
            T, B, 2, self.num_heads, C // self.num_heads, N
        )
        x = self.attn_network(qk)
        x = self.proj_network(x)  # [T, B, C, N]
        return x.reshape(T, B, C, H, W).contiguous()  # [T, B, C, H, W]


class Block(nn.Module):

    def __init__(
        self,
        neuron_type,
        attn_type: str,
        dim,
        num_heads,
        mlp_ratio=4.,
        **kwargs
    ):
        super().__init__()

        if attn_type == "SSA":
            self.attn = SSA(neuron_type, dim, num_heads=num_heads, **kwargs)
        elif attn_type == "QKA":
            self.attn = QKA(neuron_type, dim, num_heads=num_heads, **kwargs)

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


class Conv2dBNMaxPoolNeuron(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding,
        neuron_type, **kwargs
    ):
        super().__init__()
        self.conv_bn_pool = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            BatchNorm2d_(out_channels),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.neuron = get_neuron(neuron_type, **kwargs)

    def forward(self, x):
        x = self.conv_bn_pool(x)
        return self.neuron(x)

    def __spatial_split__(self):
        return self.conv_bn_pool, self.neuron


class PatchEmbedInit(nn.Module):

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
        patch_size = (patch_size, patch_size
                     ) if isinstance(patch_size, int) else patch_size
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

        self.proj_conv_0 = Conv2dBNMaxPoolNeuron(
            in_channels,
            embed_dims // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_1 = Conv2dBNMaxPoolNeuron(
            embed_dims // 2,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_2 = Conv2dBNNeuron(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_res = Conv2dBNNeuron(
            embed_dims // 2,
            embed_dims,
            kernel_size=1,
            stride=2,
            padding=0,
            neuron_type=neuron_type,
            **kwargs
        )  # residual connection as positional embedding!!

        self.proj_conv_0.x_compressor = "NullSpikeCompressor"

    def forward(self, x):
        # x is a float tensor
        x = self.proj_conv_0(x)
        x_feat = x
        x = self.proj_conv_1(x)
        x = self.proj_conv_2(x)

        x_feat = self.proj_res(x_feat)
        x = x + x_feat
        return x  # non-binary int-valued tensor


class PatchEmbedStage(nn.Module):

    def __init__(
        self,
        neuron_type,
        img_size_h=128,
        img_size_w=128,
        patch_size=4,
        embed_dims=256,
        **kwargs
    ):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = (patch_size, patch_size
                     ) if isinstance(patch_size, int) else patch_size
        if len(patch_size) != 2:
            raise ValueError(
                f"patch_size should be a tuple of length 2 or an int, "
                f"but got {len(patch_size)}"
            )
        self.patch_size = patch_size
        self.H = self.image_size[0] // self.patch_size[0]
        self.W = self.image_size[1] // self.patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv_1 = Conv2dBNMaxPoolNeuron(
            embed_dims // 2,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_conv_2 = Conv2dBNNeuron(
            embed_dims,
            embed_dims,
            kernel_size=3,
            stride=1,
            padding=1,
            neuron_type=neuron_type,
            **kwargs
        )
        self.proj_res = Conv2dBNNeuron(
            embed_dims // 2,
            embed_dims,
            kernel_size=1,
            stride=2,
            padding=0,
            neuron_type=neuron_type,
            **kwargs
        )  # residual connection as positional embedding!!

        self.proj_conv_1.x_compressor = "Uint8SpikeCompressor"
        self.proj_res.x_compressor = "Uint8SpikeCompressor"

    def forward(self, x):
        # x is a non-binary tensor
        x_feat = x
        x = self.proj_conv_1(x)
        x = self.proj_conv_2(x)

        x_feat = self.proj_res(x_feat)  # non-binary int-valued tensor
        x = x + x_feat
        return x  # non-binary int-valued tensor


class QKFormer(nn.Module):

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
        depths=10,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = T
        kwargs["T"] = T

        self.patch_embed1 = PatchEmbedInit(
            neuron_type,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims // 4,
            **kwargs
        )
        self.block1 = nn.ModuleList([
            Block(
                neuron_type,
                attn_type="QKA",
                dim=embed_dims // 4,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                **kwargs
            ) for _ in range(1)
        ])

        self.patch_embed2 = PatchEmbedStage(
            neuron_type,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            embed_dims=embed_dims // 2,
            **kwargs
        )
        self.block2 = nn.ModuleList([
            Block(
                neuron_type,
                attn_type="QKA",
                dim=embed_dims // 2,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                **kwargs
            ) for _ in range(2)
        ])

        self.patch_embed3 = PatchEmbedStage(
            neuron_type,
            img_size_h=img_size_h,
            img_size_w=img_size_w,
            patch_size=patch_size,
            embed_dims=embed_dims,
            **kwargs
        )
        self.block3 = nn.ModuleList([
            Block(
                neuron_type,
                attn_type="SSA",
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                **kwargs
            ) for _ in range(depths - 3)
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
        x = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x)  # [T, B, C, H, W]

        x = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)

        x = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x  # [B, num_classes]


def GCQKFormer(neuron_type, compress_x, level, **kwargs):
    net = QKFormer(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (
            Conv1dBNNeuron, Conv2dBNNeuron, Conv2dBNMaxPoolNeuron, QKACore,
            SSACore
        ),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )
