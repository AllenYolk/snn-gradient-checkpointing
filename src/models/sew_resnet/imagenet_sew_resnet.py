import torch
import torch.nn as nn
from spikingjelly.activation_based import layer, functional

from ..blocks import get_block, neuron_type_to_str
from ..neuron import get_neuron
from ..compress import *


def _conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )


def _conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=False
    )


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
        norm_layer=nn.BatchNorm2d,
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

        self.conv1 = layer.SeqToANNContainer(
            _conv3x3(in_planes, planes, stride), norm_layer(planes)
        )
        self.sn1 = get_neuron(neuron_type, **kwargs)

        self.conv2 = layer.SeqToANNContainer(
            _conv3x3(planes, planes), norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = get_neuron(neuron_type, **kwargs)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))

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
        norm_layer=nn.BatchNorm2d,
        **kwargs,  # neuronal parameters
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width/64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            _conv1x1(in_planes, width), norm_layer(width)
        )
        self.sn1 = get_neuron(neuron_type, **kwargs)

        self.conv2 = layer.SeqToANNContainer(
            _conv3x3(width, width, stride, groups, dilation), norm_layer(width)
        )
        self.sn2 = get_neuron(neuron_type, **kwargs)

        self.conv3 = layer.SeqToANNContainer(
            _conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = get_neuron(neuron_type, **kwargs)

    def forward(self, x):
        identity = x

        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = identity + out

        return out


def _zero_init_blocks(net: nn.Module):
    for m in net.modules():
        if isinstance(m, Bottleneck):
            nn.init.constant_(m.conv3.module[1].weight, 0)
        elif isinstance(m, BasicBlock):
            nn.init.constant_(m.conv2.module[1].weight, 0)


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
        norm_layer=nn.BatchNorm2d,
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        self._norm_layer = norm_layer
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

        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.in_planes)
        self.sn1 = get_neuron(neuron_type, **kwargs)
        self.maxpool = layer.MaxPool2d(
            kernel_size=3, stride=2, padding=1, step_mode="m"
        )

        self.layer1 = self._make_layer(
            neuron_type, block, 64, layers[0], **kwargs
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
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
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
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    _conv1x1(self.in_planes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                get_neuron(neuron_type, **kwargs),
            )

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
                norm_layer,
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
                    norm_layer=norm_layer,
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x.shape = [B, C, H, W]
        x = self.conv1(x)
        x = self.bn1(x)  # [B, C, H, W]
        x = x.repeat(self.T, 1, 1, 1, 1)
        x = self.sn1(x)  # [T, B, C, H, W]
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)  # [T, B, D]
        return self.fc(x.mean(dim=0))

    def forward(self, x):
        return self._forward_impl(x)


def sew_resnet18(neuron_type, **kwargs):
    return SEWResNet(neuron_type, BasicBlock, [2, 2, 2, 2], **kwargs)


def sew_resnet34(neuron_type, **kwargs):
    return SEWResNet(neuron_type, BasicBlock, [3, 4, 6, 3], **kwargs)


def sew_resnet50(neuron_type, **kwargs):
    return SEWResNet(neuron_type, Bottleneck, [3, 4, 6, 3], **kwargs)


def sew_resnet101(neuron_type, **kwargs):
    return SEWResNet(neuron_type, Bottleneck, [3, 4, 23, 3], **kwargs)


def sew_resnet152(neuron_type, **kwargs):
    return SEWResNet(neuron_type, Bottleneck, [3, 8, 36, 3], **kwargs)
