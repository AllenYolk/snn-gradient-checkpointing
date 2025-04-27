import sys

sys.path.append("./src")

import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.blocks import get_block, neuron_type_to_str
from modules.neuron import get_neuron
from modules.compress import *


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

        self.conv1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(
                    in_features, hidden_features, kernel_size=1, stride=1
                ), nn.BatchNorm2d(hidden_features)
            ),  # HW as patch dimensions!
            get_neuron(neuron_type, **kwargs)
        )

        self.conv2 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(
                    hidden_features, out_features, kernel_size=1, stride=1
                ), nn.BatchNorm2d(out_features)
            ), get_neuron(neuron_type, **kwargs)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MLPCheckpointing(nn.Module):

    def __init__(
        self,
        neuron_type,
        spike_compressor: str,
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

        spike_compressor_class = get_spike_compressor(spike_compressor)
        qkv_forced_uint8 = spike_compressor_class.requires_strictly_binary
        conv1_spike_compressor = (
            "Uint8SpikeCompressor" if qkv_forced_uint8 else spike_compressor
        )

        self.conv1 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv2d(
                in_features, hidden_features, kernel_size=1, stride=1
            ),
            bn=nn.BatchNorm2d(hidden_features),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(conv1_spike_compressor),
        )

        self.conv2 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv2d(
                hidden_features, out_features, kernel_size=1, stride=1
            ),
            bn=nn.BatchNorm2d(out_features),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

    def forward(self, x):
        # x is a non-binary int-valued tensor
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

        self.q_network = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(dim)
            ),
            get_neuron(neuron_type, **kwargs),
        )
        self.k_network = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(dim)
            ),
            get_neuron(neuron_type, **kwargs),
        )
        self.v_network = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(dim)
            ),
            get_neuron(neuron_type, **kwargs),
        )

        self.attn_neuron = get_neuron(neuron_type, **kwargs)

        self.proj_network = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv1d(dim, dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(dim)
            ),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x):
        # x.shape = [T, B, C, H, W]
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(3)
        T, B, C, N = x.shape

        q_conv_out = self.q_network(x)
        q = q_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        k_conv_out = self.k_network(x)
        k = k_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()

        v_conv_out = self.v_network(x)
        v = v_conv_out.transpose(-1, -2).reshape(
            T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 3, 2, 4).contiguous()  # [T, B, h, N, C//h]

        x = k.transpose(-2, -1) @ v
        x = (q@x) * self.scale  # [T, B, h, N, C//h]
        x = x.transpose(-1, -2).reshape(T, B, C, N).contiguous()
        x = self.attn_neuron(x)  # [T, B, C, N]

        x = self.proj_network(x)
        return x.reshape(T, B, C, H, W).contiguous()  # [T, B, C, H, W]


class SSACheckpointing(nn.Module):

    def __init__(
        self, neuron_type, spike_compressor: str, dim, num_heads=8, **kwargs
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"dim {dim} should be divided by num_heads {num_heads}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125

        spike_compressor_class = get_spike_compressor(spike_compressor)
        qkv_forced_uint8 = spike_compressor_class.requires_strictly_binary
        qkv_spike_compressor = (
            "Uint8SpikeCompressor" if qkv_forced_uint8 else spike_compressor
        )

        self.q_network = get_block(
            f"Conv1dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            bn=nn.BatchNorm1d(dim),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(qkv_spike_compressor),
        )
        self.k_network = get_block(
            f"Conv1dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            bn=nn.BatchNorm1d(dim),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(qkv_spike_compressor),
        )
        self.v_network = get_block(
            f"Conv1dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            bn=nn.BatchNorm1d(dim),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(qkv_spike_compressor),
        )

        self.attn_network = get_block(
            f"SSACore{neuron_type_to_str(neuron_type)}",
            scale=self.scale,
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

        self.proj_network = get_block(
            f"Conv1dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv1d(dim, dim, kernel_size=1, stride=1),
            bn=nn.BatchNorm1d(dim),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

    def forward(self, x):
        # x.shape = [T, B, C, H, W]; non-binary int-valued tensor
        H, W = x.shape[-2], x.shape[-1]
        x = x.flatten(3)
        T, B, C, N = x.shape

        qkv = torch.empty([3, T, B, C, N], device=x.device, dtype=x.dtype)
        qkv[0] = self.q_network(x)
        qkv[1] = self.k_network(x)
        qkv[2] = self.v_network(x)
        qkv = qkv.transpose(-1, -2).reshape(
            3, T, B, N, self.num_heads, C // self.num_heads
        ).permute(0, 1, 2, 4, 3, 5).contiguous()  # [3, T, B, h, N, C//h]; bi

        x = self.attn_network(qkv)  # [T, B, C, N]
        x = self.proj_network(x)
        return x.reshape(T, B, C, H, W).contiguous()  # [T, B, C, H, W]


class Block(nn.Module):

    def __init__(
        self,
        neuron_type,
        dim,
        num_heads,
        mlp_ratio=4.,
        checkpointing=False,
        spike_compressor: str = "NullSpikeCompressor",
        **kwargs
    ):
        super().__init__()

        self.attn = SSACheckpointing(
            neuron_type,
            spike_compressor,
            dim,
            num_heads=num_heads,
            **kwargs,
        ) if checkpointing else SSA(
            neuron_type,
            dim,
            num_heads=num_heads,
            **kwargs,
        )
        self.mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLPCheckpointing(
            neuron_type,
            spike_compressor,
            in_features=dim,
            hidden_features=self.mlp_hidden_dim,
            out_features=dim,
            **kwargs
        ) if checkpointing else MLP(
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

        self.proj_conv_0 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(
                    in_channels,
                    embed_dims // 8,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(embed_dims // 8),
            ),
            get_neuron(neuron_type, **kwargs),
        )

        self.proj_conv_1 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=1,
                    ceil_mode=False
                ),
                nn.Conv2d(
                    embed_dims // 8,
                    embed_dims // 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(embed_dims // 4),
            ),
            get_neuron(neuron_type, **kwargs),
        )

        self.proj_conv_2 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=1,
                    ceil_mode=False
                ),
                nn.Conv2d(
                    embed_dims // 4,
                    embed_dims // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(embed_dims // 2),
            ),
            get_neuron(neuron_type, **kwargs),
        )

        self.proj_conv_3 = nn.Sequential(
            layer.SeqToANNContainer(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=1,
                    ceil_mode=False
                ),
                nn.Conv2d(
                    embed_dims // 2,
                    embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(embed_dims),
            ),
            get_neuron(neuron_type, **kwargs),
            layer.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                ceil_mode=False,
                step_mode="m"
            ),
        )

        self.positional_encoding = nn.Sequential(
            layer.SeqToANNContainer(
                nn.Conv2d(
                    embed_dims,
                    embed_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(embed_dims),
            ),
            get_neuron(neuron_type, **kwargs),
        )  # conv as learnable positional encoding

    def forward(self, x):
        x = self.proj_conv_0(x)
        x = self.proj_conv_1(x)
        x = self.proj_conv_2(x)
        x = self.proj_conv_3(x)

        x_feat = x
        x = self.positional_encoding(x)
        x = x + x_feat
        return x


class SPSCheckpointing(nn.Module):

    def __init__(
        self,
        neuron_type,
        spike_compressor: str,
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

        self.proj_conv_0 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                in_channels,
                embed_dims // 8,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            bn=nn.BatchNorm2d(embed_dims // 8),
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            spike_compressor=get_spike_compressor("NullSpikeCompressor"),
        )
        self.proj_conv_1 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                embed_dims // 8,
                embed_dims // 4,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            bn=nn.BatchNorm2d(embed_dims // 4),
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            spike_compressor=get_spike_compressor(spike_compressor),
        )
        self.proj_conv_2 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                embed_dims // 4,
                embed_dims // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            bn=nn.BatchNorm2d(embed_dims // 2),
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            spike_compressor=get_spike_compressor(spike_compressor),
        )
        self.proj_conv_3 = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                embed_dims // 2,
                embed_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            bn=nn.BatchNorm2d(embed_dims),
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
            ),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

        self.positional_encoding = get_block(
            f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=nn.Conv2d(
                embed_dims,
                embed_dims,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            bn=nn.BatchNorm2d(embed_dims),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor)
        )

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
                checkpointing=False,
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
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x  # [B, num_classes]


class MESpikformer(nn.Module):

    def __init__(
        self,
        neuron_type,
        spike_compressor: str,
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

        self.patch_embed = SPSCheckpointing(
            neuron_type,
            spike_compressor,
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
                checkpointing=True,
                spike_compressor=spike_compressor,
                **kwargs
            ) for i in range(depths)
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
        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x  # [B, num_classes]
