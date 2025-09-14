import sys

sys.path.append("./src")

import torch
import torch.nn as nn
from spikingjelly.activation_based import layer

from modules.neuron import get_neuron
from modules.checkpointing import memory_optimization
from modules.bn import BatchNorm2d_
from modules.merge_split import RepeatT


class SeqToANNContainer(layer.SeqToANNContainer):
    """Stateless layer container that supports temporal chunking"""

    def __tc_init_states__(self, x_seq):
        return []

    def __tc_forward__(self, xc):
        return (self.forward(xc),)


class Conv3x3(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        groups=1,
        dilation=1,
        neuron_type="SJLIF",
        **kwargs
    ):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                groups=groups,
                dilation=dilation,
                bias=False
            )
        )
        self.bn_neuron = nn.Sequential(
            SeqToANNContainer(BatchNorm2d_(out_channels)),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class Conv1x1(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        neuron_type="SJLIF",
        **kwargs
    ):
        super().__init__()
        self.conv = SeqToANNContainer(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            ),
        )
        self.bn_neuron = nn.Sequential(
            SeqToANNContainer(BatchNorm2d_(out_channels)),
            get_neuron(neuron_type, **kwargs),
        )

    def forward(self, x_seq):
        return self.bn_neuron(self.conv(x_seq))

    def __spatial_split__(self):
        return self.conv, self.bn_neuron


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        neuron_type,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        **kwargs  # neuronal parameters
    ):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'SpikingBasicBlock only supports groups=1 and base_width=64'
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in SpikingBasicBlock"
            )
        self.stride = stride

        self.conv1 = Conv3x3(
            in_planes, planes, stride, neuron_type=neuron_type, **kwargs
        )
        self.conv2 = Conv3x3(planes, planes, neuron_type=neuron_type, **kwargs)
        self.downsample = downsample
        self.conv1.x_compressor = "Uint8SpikeCompressor"

    def forward(self, x):
        identity = x
        out = self.conv2(self.conv1(x))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + out
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        neuron_type,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        self.stride = stride
        width = int(planes * (base_width/64.)) * groups
        self.width = width

        self.conv1 = Conv1x1(
            in_planes, width, neuron_type=neuron_type, **kwargs
        )
        self.conv2 = Conv3x3(
            width,
            width,
            stride,
            groups,
            dilation,
            neuron_type=neuron_type,
            **kwargs
        )
        self.conv3 = Conv1x1(
            width, planes * self.expansion, neuron_type=neuron_type, **kwargs
        )
        self.downsample = downsample
        self.conv1.x_compressor = "Uint8SpikeCompressor"

    def forward(self, x):
        identity = x
        out = self.conv3(self.conv2(self.conv1(x)))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = identity + out
        return out


def _zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.bn_neuron.module[0].weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.bn_neuron.module[0].weight, 0)


class PreConv(nn.Module):

    def __init__(self, C_in, planes, T, neuron_type, **kwargs):
        super().__init__()
        self.T = T
        kwargs["T"] = T
        self.conv_bn = nn.Sequential(
            nn.Conv2d(
                C_in,
                planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            BatchNorm2d_(planes),
        )
        self.neuron_pool = nn.Sequential(
            RepeatT(T),
            get_neuron(neuron_type, **kwargs),
            SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return self.neuron_pool(x)

    def __spatial_split__(self):
        return self.conv_bn, self.neuron_pool


class SEWResNet(nn.Module):

    def __init__(
        self,
        neuron_type,
        block,
        layers,
        T=4,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        kwargs["T"] = T  # for PSN
        self.T = T
        self.in_planes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".
                format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        self.pre_conv = PreConv(
            3, self.in_planes, neuron_type=neuron_type, **kwargs
        )

        self.layer1 = self._make_layer(
            neuron_type,
            block,
            64,
            layers[0],
            **kwargs,
        )
        self.layer2 = self._make_layer(
            neuron_type,
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            **kwargs,
        )
        self.layer3 = self._make_layer(
            neuron_type,
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            **kwargs,
        )
        self.layer4 = self._make_layer(
            neuron_type,
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            **kwargs,
        )

        self.avgpool = SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            _zero_init_blocks(self)

    def _make_layer(
        self,
        neuron_type,
        block,
        planes,
        blocks,
        stride=1,
        dilate=False,
        **kwargs,
    ):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = Conv1x1(
                self.in_planes,
                planes * block.expansion,
                stride,
                neuron_type=neuron_type,
                **kwargs
            )
            downsample.x_compressor = "Uint8SpikeCompressor"

        layers = []
        layers.append(
            block(
                neuron_type,
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                **kwargs,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    neuron_type,
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # x.shape = [B, C, H, W]
        x = self.pre_conv(x)  # [T, B, C, H, W]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)  # [T, B, D]
        return self.fc(x)  # [T, B, num_classes]


class SEWResNet18(SEWResNet):

    def __init__(self, neuron_type, **kwargs):
        super().__init__(neuron_type, BasicBlock, [2, 2, 2, 2], **kwargs)


class SEWResNet34(SEWResNet):

    def __init__(self, neuron_type, **kwargs):
        super().__init__(neuron_type, BasicBlock, [3, 4, 6, 3], **kwargs)


class SEWResNet50(SEWResNet):

    def __init__(self, neuron_type, **kwargs):
        super().__init__(neuron_type, Bottleneck, [3, 4, 6, 3], **kwargs)


class SEWResNet101(SEWResNet):

    def __init__(self, neuron_type, **kwargs):
        super().__init__(neuron_type, Bottleneck, [3, 4, 23, 3], **kwargs)


class SEWResNet152(SEWResNet):

    def __init__(self, neuron_type, **kwargs):
        super().__init__(neuron_type, Bottleneck, [3, 8, 36, 3], **kwargs)


def GCSEWResNet18(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet18(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3, PreConv),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )


def GCSEWResNet34(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet34(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3, PreConv),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )


def GCSEWResNet50(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet50(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3, PreConv),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )


def GCSEWResNet101(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet101(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3, PreConv),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )


def GCSEWResNet152(neuron_type, compress_x, level, **kwargs):
    net = SEWResNet152(neuron_type, **kwargs)
    return memory_optimization(
        net,
        (Conv1x1, Conv3x3, PreConv),
        dummy_input=torch.zeros(32, 3, 224, 224) + 0.9,
        compress_x=compress_x,
        level=level,
        verbose=True,
    )
