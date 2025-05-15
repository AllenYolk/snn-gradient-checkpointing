import sys

sys.path.append("./src")

import torch
import torch.nn as nn
import torchvision
from spikingjelly.activation_based import layer

from modules.blocks import get_block, neuron_type_to_str
from modules.neuron import get_neuron
from modules.compress import *
from modules.merge_split import RepeatT


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
            _conv3x3(in_planes, planes, stride),
            norm_layer(planes),
        )
        self.sn1 = get_neuron(neuron_type, **kwargs)

        self.conv2 = layer.SeqToANNContainer(
            _conv3x3(planes, planes),
            norm_layer(planes),
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


class BasicBlockCheckpointing(nn.Module):
    expansion = 1

    def __init__(
        self,
        neuron_type,
        spike_compressor: str,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
        input_non_binary_int: bool = True,
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
        if norm_layer != nn.BatchNorm2d:
            raise NotImplementedError(
                "Only nn.BatchNorm2d is supported in BasicBlockCheckpointing"
            )
        spike_compressor_class = get_spike_compressor(spike_compressor)
        forced_uint8 = (
            spike_compressor_class.requires_strictly_binary and
            input_non_binary_int
        )

        self.conv_bn_sn1 = get_block(
            block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=_conv3x3(in_planes, planes, stride),
            bn=norm_layer(planes),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(
                "Uint8SpikeCompressor" if forced_uint8 else spike_compressor
            ),
        )

        self.conv_bn_sn2 = get_block(
            block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=_conv3x3(planes, planes),
            bn=norm_layer(planes),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_bn_sn1(x)
        out = self.conv_bn_sn2(out)

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
        super().__init__()
        width = int(planes * (base_width/64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            _conv1x1(in_planes, width), norm_layer(width)
        )
        self.sn1 = get_neuron(neuron_type, **kwargs)

        self.conv2 = layer.SeqToANNContainer(
            _conv3x3(width, width, stride, groups, dilation),
            norm_layer(width),
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


class BottleneckCheckpointing(nn.Module):
    expansion = 4

    def __init__(
        self,
        neuron_type,
        spike_compressor: str,
        in_planes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
        input_non_binary_int: bool = True,
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        width = int(planes * (base_width/64.)) * groups
        spike_compressor_class = get_spike_compressor(spike_compressor)
        forced_uint8 = (
            spike_compressor_class.requires_strictly_binary and
            input_non_binary_int
        )

        self.conv_bn_sn1 = get_block(
            block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=_conv1x1(in_planes, width),
            bn=norm_layer(width),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(
                "Uint8SpikeCompressor" if forced_uint8 else spike_compressor
            ),
        )

        self.conv_bn_sn2 = get_block(
            block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=_conv3x3(width, width, stride, groups, dilation),
            bn=norm_layer(width),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

        self.conv_bn_sn3 = get_block(
            block_type=f"Conv2dBN{neuron_type_to_str(neuron_type)}",
            proj=_conv1x1(width, planes * self.expansion),
            bn=norm_layer(planes * self.expansion),
            neuron=get_neuron(neuron_type, **kwargs),
            spike_compressor=get_spike_compressor(spike_compressor),
        )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_bn_sn1(x)
        out = self.conv_bn_sn2(out)
        out = self.conv_bn_sn3(out)

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
        spike_compressor="IdentitySpikeCompressor",
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        kwargs["T"] = T  # for PSN
        self._norm_layer = norm_layer
        self.T = T
        self.in_planes = 64
        self.dilation = 1
        checkpointing = block.__name__.endswith("Checkpointing")
        self.checkpointing = checkpointing
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

        self.pre_conv = get_block(
            f"Conv2dBNRepeat{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                3,
                self.in_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            bn=norm_layer(self.in_planes),
            T=T,
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            spike_compressor=get_spike_compressor("NullSpikeCompressor"),
        ) if checkpointing else nn.Sequential(
            nn.Conv2d(
                3,
                self.in_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            norm_layer(self.in_planes),
            RepeatT(T),
            get_neuron(neuron_type, **kwargs),
            layer.MaxPool2d(kernel_size=3, stride=2, padding=1, step_mode="m"),
        )

        self.layer1 = self._make_layer(
            neuron_type,
            block,
            64,
            layers[0],
            checkpointing=checkpointing,
            spike_compressor=spike_compressor,
            input_non_binary_int=False,  # input to its first res block is binary!
            **kwargs
        )
        self.layer2 = self._make_layer(
            neuron_type,
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            checkpointing=checkpointing,
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
            **kwargs,
        )
        self.layer3 = self._make_layer(
            neuron_type,
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            checkpointing=checkpointing,
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
            **kwargs,
        )
        self.layer4 = self._make_layer(
            neuron_type,
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            checkpointing=checkpointing,
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
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
        checkpointing=False,
        spike_compressor="IdentitySpikeCompressor",
        input_non_binary_int: bool = True,
        **kwargs,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        spike_compressor_class = get_spike_compressor(spike_compressor)
        downsample_forced_uint8 = (
            spike_compressor_class.requires_strictly_binary and
            input_non_binary_int
        )
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = get_block(
                f"Conv2dBN{neuron_type_to_str(neuron_type)}",
                proj=_conv1x1(self.in_planes, planes * block.expansion, stride),
                bn=norm_layer(planes * block.expansion),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(
                    "Uint8SpikeCompressor"
                    if downsample_forced_uint8 else spike_compressor
                ),
            ) if checkpointing else nn.Sequential(
                layer.SeqToANNContainer(
                    _conv1x1(self.in_planes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
                get_neuron(neuron_type, **kwargs),
            )

        layers = []
        neuron_type = [neuron_type]
        if checkpointing:
            neuron_type.append(spike_compressor)
        layers.append(
            block(
                *neuron_type,
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                input_non_binary_int=input_non_binary_int,
                **kwargs,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    *neuron_type,
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    forced_uint8=True,  # input cannot be binary
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x.shape = [B, C, H, W]
        x = self.pre_conv(x)  # [T, B, C, H, W]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)  # [T, B, D]
        return self.fc(x)  # [T, B, num_classes]

    def forward(self, x):
        return self._forward_impl(x)


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


class FGCSEWResNet18(SEWResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            BasicBlockCheckpointing, [2, 2, 2, 2],
            spike_compressor=spike_compressor,
            **kwargs
        )


class FGCSEWResNet34(SEWResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            BasicBlockCheckpointing, [3, 4, 6, 3],
            spike_compressor=spike_compressor,
            **kwargs
        )


class FGCSEWResNet50(SEWResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            BottleneckCheckpointing, [3, 4, 6, 3],
            spike_compressor=spike_compressor,
            **kwargs
        )


class FGCSEWResNet101(SEWResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            BottleneckCheckpointing, [3, 4, 23, 3],
            spike_compressor=spike_compressor,
            **kwargs
        )


class FGCSEWResNet152(SEWResNet):

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        super().__init__(
            neuron_type,
            BottleneckCheckpointing, [3, 8, 36, 3],
            spike_compressor=spike_compressor,
            **kwargs
        )


class PGCSEWResNet(nn.Module):

    def __init__(
        self,
        neuron_type,
        block_types,
        layers,
        T=4,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=nn.BatchNorm2d,
        spike_compressor="IdentitySpikeCompressor",
        **kwargs,  # neuronal parameters
    ):
        super().__init__()
        kwargs["T"] = T  # for PSN
        self._norm_layer = norm_layer
        self.T = T
        self.in_planes = 64
        self.dilation = 1
        checkpointing = self._get_checkpointing(block_types)
        self.checkpointing = checkpointing

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

        self.pre_conv = get_block(
            f"Conv2dBNRepeat{neuron_type_to_str(neuron_type)}MaxPool2d",
            proj=nn.Conv2d(
                3,
                self.in_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            bn=norm_layer(self.in_planes),
            T=T,
            neuron=get_neuron(neuron_type, **kwargs),
            pool=nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            spike_compressor=get_spike_compressor("NullSpikeCompressor"),
        )

        self.layer1 = self._make_layer(
            neuron_type,
            block_types[0],
            64,
            layers[0],
            checkpointing=checkpointing[0],
            spike_compressor=spike_compressor,
            input_non_binary_int=False,  # input to its first res block is binary!
            **kwargs
        )
        self.layer2 = self._make_layer(
            neuron_type,
            block_types[1],
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
            checkpointing=checkpointing[1],
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
            **kwargs,
        )
        self.layer3 = self._make_layer(
            neuron_type,
            block_types[2],
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
            checkpointing=checkpointing[2],
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
            **kwargs,
        )
        self.layer4 = self._make_layer(
            neuron_type,
            block_types[3],
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
            checkpointing=checkpointing[3],
            spike_compressor=spike_compressor,
            input_non_binary_int=True,
            **kwargs,
        )
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1), step_mode="m")
        self.fc = nn.Linear(512 * block_types[3][-1].expansion, num_classes)

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

    @staticmethod
    def _get_checkpointing(block_types):
        c = []
        for bt in block_types:
            cc = []
            for b in bt:
                cc.append(b.__name__.endswith("Checkpointing"))
            c.append(cc)
        return c

    def _make_layer(
        self,
        neuron_type,
        block_types,
        planes,
        blocks,
        stride=1,
        dilate=False,
        checkpointing=[False],
        spike_compressor="IdentitySpikeCompressor",
        input_non_binary_int: bool = True,
        **kwargs,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        spike_compressor_class = get_spike_compressor(spike_compressor)
        downsample_forced_uint8 = (
            spike_compressor_class.requires_strictly_binary and
            input_non_binary_int
        )
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.in_planes != planes * block_types[0].expansion:
            downsample = get_block(
                f"Conv2dBN{neuron_type_to_str(neuron_type)}",
                proj=_conv1x1(
                    self.in_planes, planes * block_types[0].expansion, stride
                ),
                bn=norm_layer(planes * block_types[0].expansion),
                neuron=get_neuron(neuron_type, **kwargs),
                spike_compressor=get_spike_compressor(
                    "Uint8SpikeCompressor"
                    if downsample_forced_uint8 else spike_compressor
                ),
            ) if checkpointing[0] else nn.Sequential(
                layer.SeqToANNContainer(
                    _conv1x1(
                        self.in_planes, planes *
                        block_types[0].expansion, stride
                    ),
                    norm_layer(planes * block_types[0].expansion),
                ),
                get_neuron(neuron_type, **kwargs),
            )

        layers = []
        ntsc = [neuron_type]
        if checkpointing[0]:
            ntsc.append(spike_compressor)
        layers.append(
            block_types[0](
                *ntsc,
                self.in_planes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
                input_non_binary_int=input_non_binary_int,
                **kwargs,
            )
        )
        self.in_planes = planes * block_types[0].expansion
        for i in range(1, blocks):
            ntsc = [neuron_type]
            if checkpointing[i]:
                ntsc.append(spike_compressor)
            layers.append(
                block_types[i](
                    *ntsc,
                    self.in_planes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    forced_uint8=True,  # input cannot be binary
                    **kwargs
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # x.shape = [B, C, H, W]
        x = self.pre_conv(x)  # [T, B, C, H, W]

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 2)  # [T, B, D]
        return self.fc(x)  # [T, B, num_classes]

    def forward(self, x):
        return self._forward_impl(x)


class PGCSEWResNet34(PGCSEWResNet):

    layers = [3, 4, 6, 3]

    def __init__(self, neuron_type, spike_compressor, **kwargs):
        l = self.layers
        super().__init__(
            neuron_type,
            block_types=[
                [BasicBlock] + [BasicBlockCheckpointing] * (l[0] - 1),
                [BasicBlock] + [BasicBlockCheckpointing] * (l[1] - 1),
                [BasicBlock] + [BasicBlockCheckpointing] * (l[2] - 1),
                [BasicBlock] * l[3],
            ],
            layers=self.layers,
            spike_compressor=spike_compressor,
            **kwargs
        )


def ResNet34(**kwargs):
    return torchvision.models.resnet34()
