import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn
import einops

from models import SJLIFNode, HandWrittenLIFNode
from models import Conv2dLIF
from models import IdentitySpikeCompressor, BitSpikeCompressor
from models import BooleanSpikeCompressor, SparseSpikeCompressor
from models import Uint8SpikeCompressor


def make_parameters_equal(net, reference_net):
    """Assume that `net` and `reference_net` have the parameter order.
    """
    for p, ref_p in zip(net.parameters(), reference_net.parameters()):
        p.data = ref_p.data.clone()


class MergeTN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_seq):
        return einops.rearrange(x_seq, "T N ... -> (T N) ...")


class SplitTN(nn.Module):

    def __init__(self, T):
        super().__init__()
        self.T = T

    def forward(self, x_seq):
        return einops.rearrange(x_seq, "(T N) ... -> T N ...", T=self.T)


def test_compressor_equality():
    T = 20
    N = 64
    C = 48
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = (x >= 0.0).float()
    grad_s = torch.randn(T, N, C, H, W)

    net1 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 5, padding=2, bias=True),
        neuron=HandWrittenLIFNode(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net11 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 5, padding=2, bias=True),
        neuron=SJLIFNode(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 5, padding=2, bias=True),
        neuron=HandWrittenLIFNode(decay_lambda=0.5),
        spike_compressor=BooleanSpikeCompressor()
    )
    net3 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 5, padding=2, bias=True),
        neuron=HandWrittenLIFNode(decay_lambda=0.5),
        spike_compressor=SparseSpikeCompressor(dtype=torch.int64)
    )
    net4 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 5, padding=2, bias=True),
        neuron=HandWrittenLIFNode(decay_lambda=0.5),
        spike_compressor=BitSpikeCompressor()
    )
    make_parameters_equal(net11, net1)
    make_parameters_equal(net2, net1)
    make_parameters_equal(net3, net1)
    make_parameters_equal(net4, net1)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    s1.backward(grad_s)

    x11 = x.clone()
    x11.requires_grad = True
    s11 = net11(x11)
    s11.backward(grad_s)

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    s2.backward(grad_s)

    x3 = x.clone()
    x3.requires_grad = True
    s3 = net3(x3)
    s3.backward(grad_s)

    x4 = x.clone()
    x4.requires_grad = True
    s4 = net4(x4)
    s4.backward(grad_s)

    assert torch.allclose(s1, s11, atol=1e-5)
    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(s1, s3, atol=1e-5)
    assert torch.allclose(s1, s4, atol=1e-5)
    assert torch.allclose(
        net1.proj.weight.grad, net11.proj.weight.grad, atol=1e-5
    )
    assert torch.allclose(
        net1.proj.weight.grad, net2.proj.weight.grad, atol=1e-5
    )
    assert torch.allclose(
        net1.proj.weight.grad, net3.proj.weight.grad, atol=1e-5
    )
    assert torch.allclose(
        net1.proj.weight.grad, net4.proj.weight.grad, atol=1e-5
    )
    assert torch.allclose(net1.proj.bias.grad, net11.proj.bias.grad, atol=1e-5)
    assert torch.allclose(net1.proj.bias.grad, net2.proj.bias.grad, atol=1e-5)
    assert torch.allclose(net1.proj.bias.grad, net3.proj.bias.grad, atol=1e-5)
    assert torch.allclose(net1.proj.bias.grad, net4.proj.bias.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x11.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x3.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x4.grad, atol=1e-5)
    print("Firing rate: ", s1.mean().item())
    print(IdentitySpikeCompressor().__class__.__name__)


def test_compressor_memory(compressor):
    T = 4
    N = 32
    C = 32
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = (x >= 0.2).float()
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(36):
        net.add_module(
            f"{i}",
            Conv2dLIF(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                neuron=HandWrittenLIFNode(decay_lambda=0.99),
                spike_compressor=compressor
            ),
        )
    net = net.to("cuda:0")

    torch.cuda.reset_peak_memory_stats("cuda:0")
    s = net(x)
    loss = s.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"{compressor.__class__.__name__}, "
        f"Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_identity_compressor_memory():
    test_compressor_memory(IdentitySpikeCompressor())


def test_boolean_compressor_memory():
    test_compressor_memory(BooleanSpikeCompressor())


def test_uint8_compressor_memory():
    test_compressor_memory(Uint8SpikeCompressor())


def test_sparse_compressor_memory():
    test_compressor_memory(SparseSpikeCompressor(dtype=torch.int64))


def test_bit_compressor_memory():
    test_compressor_memory(BitSpikeCompressor())


def test_conventional_memory():
    T = 4
    N = 32
    C = 32
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = (x >= 0.2).float()
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(36):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(
            f"{i}neuron",
            HandWrittenLIFNode(backend="torch", detach_reset=False)
        )
    net = net.to("cuda:0")

    torch.cuda.reset_peak_memory_stats("cuda:0")
    s = net(x)
    loss = s.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"Conventional, "
        f"Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


if __name__ == "__main__":
    test_compressor_equality()
    test_identity_compressor_memory()
    test_boolean_compressor_memory()
    test_uint8_compressor_memory()
    test_sparse_compressor_memory()
    test_bit_compressor_memory()
