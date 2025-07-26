import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional
from spikingjelly.activation_based.neuron import ParametricLIFNode


def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ), ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode="m")
    )


def conv1x1(in_channels, out_channels):
    return nn.Sequential(
        layer.SeqToANNContainer(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
        ), ParametricLIFNode(init_tau=2.0, detach_reset=True, step_mode="m")
    )


class SEWBlock(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        out = self.conv(x)
        out += x
        return out


class PlainBlock(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            conv3x3(mid_channels, in_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, mid_channels):
        super().__init__()
        self.conv = nn.Sequential(
            conv3x3(in_channels, mid_channels),
            layer.SeqToANNContainer(
                nn.Conv2d(
                    mid_channels,
                    in_channels,
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
            ),
        )
        self.sn = ParametricLIFNode(
            init_tau=2.0, detach_reset=True, step_mode="m"
        )

    def forward(self, x: torch.Tensor):
        return self.sn(x + self.conv(x))


class ResNetN(nn.Module):

    def __init__(self, layer_list, num_classes, connect_f=None):
        super().__init__()
        in_channels = 2
        conv = []

        for cfg_dict in layer_list:
            channels = cfg_dict['channels']

            if 'mid_channels' in cfg_dict:
                mid_channels = cfg_dict['mid_channels']
            else:
                mid_channels = channels

            if in_channels != channels:
                if cfg_dict['up_kernel_size'] == 3:
                    conv.append(conv3x3(in_channels, channels))
                elif cfg_dict['up_kernel_size'] == 1:
                    conv.append(conv1x1(in_channels, channels))
                else:
                    raise NotImplementedError

            in_channels = channels

            if 'num_blocks' in cfg_dict:
                num_blocks = cfg_dict['num_blocks']
                if cfg_dict['block_type'] == 'sew':
                    for _ in range(num_blocks):
                        conv.append(SEWBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'plain':
                    for _ in range(num_blocks):
                        conv.append(PlainBlock(in_channels, mid_channels))
                elif cfg_dict['block_type'] == 'basic':
                    for _ in range(num_blocks):
                        conv.append(BasicBlock(in_channels, mid_channels))
                else:
                    raise NotImplementedError

            if 'k_pool' in cfg_dict:
                k_pool = cfg_dict['k_pool']
                conv.append(
                    layer.SeqToANNContainer(nn.MaxPool2d(k_pool, k_pool))
                )

        conv.append(nn.Flatten(2))

        self.conv = nn.Sequential(*conv)

        with torch.no_grad():
            x = torch.zeros([1, 1, 128, 128])
            for m in self.conv.modules():
                if isinstance(m, nn.MaxPool2d):
                    x = m(x)
            out_features = x.numel() * in_channels

        self.out = nn.Linear(out_features, num_classes)

    def forward(self, x_seq: torch.Tensor):
        functional.reset_net(self)
        # x_seq.shape = [N, T, 2, H, W]
        x_seq = x_seq.permute(1, 0, 2, 3, 4)  # [T, N, 2, H, W]
        x_seq = self.conv(x_seq)
        return self.out(x_seq.mean(0))


def SEWResNet():
    layer_list = [
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
        {
            'channels': 32,
            'up_kernel_size': 1,
            'mid_channels': 32,
            'num_blocks': 1,
            'block_type': 'sew',
            'k_pool': 2
        },
    ]
    num_classes = 11
    return ResNetN(layer_list, num_classes)
