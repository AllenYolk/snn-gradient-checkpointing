import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn
import einops

from models import SJLIFNode, HandWrittenLIFNode
from models import LinearLIF, Conv2dLIF


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


def test_linear_lif_equality():
    T = 100
    N = 64
    C = 128
    x = torch.randn(T, N, C) + 0.6
    x = (x >= 0.2).float()
    grad_s = torch.randn(T, N, C)

    net1 = LinearLIF(
        proj=nn.Linear(C, C, bias=True),
        neuron=HandWrittenLIFNode(),
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SJLIFNode(backend="torch"),
    )
    make_parameters_equal(net2, net1)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    s1.backward(grad_s)

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    s2.backward(grad_s)

    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    assert torch.allclose(net1.proj.weight.grad, net2[0].weight.grad, atol=1e-5)
    assert torch.allclose(net1.proj.bias.grad, net2[0].bias.grad, atol=1e-5)


def test_linear_lif_memory_blocked():
    T = 50
    N = 64
    C = 700
    x = torch.randn(T, N, C) + 0.6
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearLIF(
                proj=nn.Linear(C, C, bias=True),
                neuron=HandWrittenLIFNode(detach_reset=False),
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
        f"Blocked Implementation, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_linear_lif_memory_conventional():
    T = 50
    N = 64
    C = 700
    x = torch.randn(T, N, C) + 0.6
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
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
        f"Conventional Implementation, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_conv2d_lif_equality():
    T = 20
    N = 64
    C = 48
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    grad_s = torch.randn(T, N, C, H, W)

    net1 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        neuron=HandWrittenLIFNode(),
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SJLIFNode(backend="torch"),
    )
    make_parameters_equal(net2, net1)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    s1.backward(grad_s)

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    s2.backward(grad_s)

    assert torch.allclose(s1, s2, atol=1e-5)
    assert torch.allclose(net1.proj.weight.grad, net2[1].weight.grad, atol=1e-5)
    assert torch.allclose(net1.proj.bias.grad, net2[1].bias.grad, atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
    print("Firing rate: ", s1.mean().item())


def test_conv2d_lif_memory_blocked():
    T = 4
    N = 32
    C = 32
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dLIF(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                neuron=HandWrittenLIFNode(detach_reset=False),
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
        f"Blocked Implementation, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_conv2d_lif_memory_conventional():
    T = 4
    N = 32
    C = 32
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = x.to("cuda:0")

    net = nn.Sequential()
    for i in range(50):
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
        f"Conventional Implementation, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


if __name__ == "__main__":
    test_linear_lif_equality()
    test_linear_lif_memory_blocked()
    test_linear_lif_memory_conventional()

    test_conv2d_lif_equality()
    test_conv2d_lif_memory_blocked()
    test_conv2d_lif_memory_conventional()
