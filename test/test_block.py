import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn

from modules import *


def make_parameters_equal(net, reference_net):
    """Assume that `net` and `reference_net` have the parameter order.
    """
    for p, ref_p in zip(net.parameters(), reference_net.parameters()):
        p.data = ref_p.data.clone()


def check_equal(x1, x2):
    assert (
        torch.allclose(x1, x2, atol=1e-5) or
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0).item() > 0.99
    ), (
        (torch.abs(x1 - x2) / (torch.abs(x1) + 1e-8)).mean(),
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0),
    )


def _test_linear_equatlity(
    net1, net2, C=128, T=100, with_bn=False, with_psn=False
):
    N = 64
    x = torch.randn(T, N, C) + 0.6
    x = (x >= 0.0).float()
    x = x.to("cuda:0")

    net1 = net1.to("cuda:0")
    net2 = net2.to("cuda:0")
    make_parameters_equal(net2, net1)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    loss1 = (s1 - 0.9).pow(2).sum()
    loss1.backward()

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    loss2 = (s2 - 0.9).pow(2).sum()
    loss2.backward()

    check_equal(s1, s2)
    if with_psn:
        idx = 4 if with_bn else 1
        check_equal(net1.neuron.weight.grad, net2[idx].weight.grad)
        check_equal(net1.neuron.bias.grad, net2[idx].bias.grad)
    check_equal(x1.grad, x2.grad)
    if with_bn:
        check_equal(net1.bn.running_mean, net2[2].running_mean)
        check_equal(net1.bn.running_var, net2[2].running_var)
        check_equal(net1.bn.weight.grad, net2[2].weight.grad)
        check_equal(net1.bn.bias.grad, net2[2].bias.grad)
    check_equal(net1.proj.weight.grad, net2[0].weight.grad)
    check_equal(net1.proj.bias.grad, net2[0].bias.grad)
    print("Firing rate: ", s1.mean().item())


def _test_conv2d_equality(
    net1, net2, C=48, T=20, with_bn=False, with_psn=False
):
    N = 64
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = (x >= 0.0).float()
    x = x.to("cuda:0")

    net1 = net1.to("cuda:0")
    net2 = net2.to("cuda:0")
    make_parameters_equal(net2, net1)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    loss1 = (s1 - 0.9).pow(2).sum()
    loss1.backward()

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    loss2 = (s2 - 0.9).pow(2).sum()
    loss2.backward()

    check_equal(s1, s2)
    check_equal(x1.grad, x2.grad)
    check_equal(net1.proj.weight.grad, net2[1].weight.grad)
    check_equal(net1.proj.bias.grad, net2[1].bias.grad)
    if with_bn:
        check_equal(net1.bn.running_mean, net2[2].running_mean)
        check_equal(net1.bn.running_var, net2[2].running_var)
        check_equal(net1.bn.weight.grad, net2[2].weight.grad)
        check_equal(net1.bn.bias.grad, net2[2].bias.grad)
    if with_psn:
        idx = 4 if with_bn else 3
        check_equal(net1.neuron.weight.grad, net2[idx].weight.grad)
        check_equal(net1.neuron.bias.grad, net2[idx].bias.grad)
    print("Firing rate: ", s1.mean().item())


def _test_linear_memory(net, C=700, T=50):
    N = 64
    x = torch.randn(T, N, C) + 0.6
    x = x.to("cuda:0")

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


def _test_conv2d_memory(net, C=32, T=4):
    N = 32
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = x.to("cuda:0")

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


def test_linear_lif_equality():
    C = 128
    T = 100
    net1 = LinearLIF(
        proj=nn.Linear(C, C, bias=True),
        neuron=HandWrittenLIF(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=False)
    _test_linear_equatlity(net1, net2, C, T, with_bn=False)


def test_linear_lif_memory_blocked():
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearLIF(
                proj=nn.Linear(C, C, bias=True),
                neuron=HandWrittenLIF(detach_reset=False),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C)


def test_linear_lif_memory_conventional():
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(
            f"{i}neuron", HandWrittenLIF(backend="torch", detach_reset=False)
        )
    _test_linear_memory(net, C)


def test_linear_PSN_equality():
    C = 128
    T = 100
    net1 = LinearPSN(
        proj=nn.Linear(C, C, bias=True),
        neuron=PSN(T=T),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        PSN(T=T),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=False, with_psn=True)
    _test_linear_equatlity(net1, net2, C, T, with_bn=False, with_psn=True)


def test_linear_PSN_memory_blocked():
    T = 50
    C = 700

    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearPSN(
                proj=nn.Linear(C, C, bias=True),
                neuron=PSN(T=T),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C, T)


def test_linear_PSN_memory_conventional():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(f"{i}neuron", PSN(T=T))
    net = net.to("cuda:0")
    _test_linear_memory(net, C, T)


def test_linear_SlidingPSN_equality():
    C = 128
    T = 100
    net1 = LinearSlidingPSN(
        proj=nn.Linear(C, C, bias=True),
        neuron=SlidingPSN(k=T // 5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SlidingPSN(k=T // 5),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=False, with_psn=True)
    _test_linear_equatlity(net1, net2, C, T, with_bn=False, with_psn=True)


def test_linear_SlidingPSN_memory_blocked():
    T = 50
    C = 700

    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearSlidingPSN(
                proj=nn.Linear(C, C, bias=True),
                neuron=SlidingPSN(k=T // 5),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C, T)


def test_linear_SlidingPSN_memory_conventional():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(f"{i}neuron", SlidingPSN(k=T // 5))
    net = net.to("cuda:0")
    _test_linear_memory(net, C, T)


def test_linear_bn_lif_equality():
    T = 100
    C = 128
    net1 = LinearBNLIF(
        proj=nn.Linear(C, C, bias=True),
        bn=nn.BatchNorm1d(C),
        neuron=HandWrittenLIF(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=True)
    _test_linear_equatlity(net1, net2, C, T, with_bn=True)


def test_linear_bn_lif_memory_blocked():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearBNLIF(
                proj=nn.Linear(C, C, bias=True),
                bn=nn.BatchNorm1d(C),
                neuron=HandWrittenLIF(detach_reset=False),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C, T)


def test_linear_bn_lif_memory_conventional():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}bn", nn.BatchNorm1d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(
            f"{i}neuron", HandWrittenLIF(backend="torch", detach_reset=False)
        )
    _test_linear_memory(net, C, T)


def test_linear_bn_PSN_equality():
    C = 128
    T = 100
    net1 = LinearBNPSN(
        proj=nn.Linear(C, C, bias=True),
        bn=nn.BatchNorm1d(C),
        neuron=PSN(T=T),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=True, with_psn=True)
    _test_linear_equatlity(net1, net2, C, T, with_bn=True, with_psn=True)


def test_linear_bn_PSN_memory_blocked():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearBNPSN(
                proj=nn.Linear(C, C, bias=True),
                bn=nn.BatchNorm1d(C),
                neuron=PSN(T=T),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C, T)


def test_linear_bn_PSN_memory_conventional():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}bn", nn.BatchNorm1d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", PSN(T=T))
    net = net.to("cuda:0")
    _test_linear_memory(net, C, T)


def test_linear_bn_SlidingPSN_equality():
    C = 128
    T = 100
    net1 = LinearBNSlidingPSN(
        proj=nn.Linear(C, C, bias=True),
        bn=nn.BatchNorm1d(C),
        neuron=SlidingPSN(k=T // 5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SlidingPSN(k=T // 5),
    )
    _test_linear_equatlity(net1, net2, C, T, with_bn=True, with_psn=True)
    _test_linear_equatlity(net1, net2, C, T, with_bn=True, with_psn=True)


def test_linear_bn_SlidingPSN_memory_blocked():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(
            f"{i}",
            LinearBNSlidingPSN(
                proj=nn.Linear(C, C, bias=True),
                bn=nn.BatchNorm1d(C),
                neuron=SlidingPSN(k=T // 5),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_linear_memory(net, C, T)


def test_linear_bn_SlidingPSN_memory_conventional():
    T = 50
    C = 700
    net = nn.Sequential()
    for i in range(100):
        net.add_module(f"{i}proj", nn.Linear(C, C, bias=True))
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}bn", nn.BatchNorm1d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", SlidingPSN(k=T // 5))
    net = net.to("cuda:0")
    _test_linear_memory(net, C, T)


def test_conv2d_lif_equality():
    C = 48
    T = 20
    net1 = Conv2dLIF(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        neuron=HandWrittenLIF(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=False)
    _test_conv2d_equality(net1, net2, C, T, with_bn=False)


def test_conv2d_lif_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dLIF(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                neuron=HandWrittenLIF(detach_reset=False),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_lif_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(
            f"{i}neuron", HandWrittenLIF(backend="torch", detach_reset=False)
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_PSN_equality():
    T = 20
    C = 48
    net1 = Conv2dPSN(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        neuron=PSN(T=T),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=False, with_psn=True)
    _test_conv2d_equality(net1, net2, C, T, with_bn=False, with_psn=True)


def test_conv2d_PSN_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dPSN(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                neuron=PSN(T=T),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_PSN_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", PSN(T=T))
    _test_conv2d_memory(net, C, T)


def test_conv2d_SlidingPSN_equality():
    T = 4
    C = 48
    net1 = Conv2dSlidingPSN(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        neuron=SlidingPSN(k=T // 2),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=False, with_psn=True)
    _test_conv2d_equality(net1, net2, C, T, with_bn=False, with_psn=True)


def test_conv2d_SlidingPSN_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dSlidingPSN(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                neuron=SlidingPSN(k=T // 2),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_SlidingPSN_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", SlidingPSN(k=T // 2))
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_lif_equality():
    T = 20
    C = 48
    net1 = Conv2dBNLIF(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        bn=nn.BatchNorm2d(C),
        neuron=HandWrittenLIF(decay_lambda=0.5),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=True)
    _test_conv2d_equality(net1, net2, C, T, with_bn=True)


def test_conv2d_bn_lif_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dBNLIF(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                bn=nn.BatchNorm2d(C),
                neuron=HandWrittenLIF(detach_reset=False),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_lif_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}bn", nn.BatchNorm2d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(
            f"{i}neuron", HandWrittenLIF(backend="torch", detach_reset=False)
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_PSN_equality():
    T = 4
    C = 15
    net1 = Conv2dBNPSN(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        bn=nn.BatchNorm2d(C),
        neuron=PSN(T=T),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=True, with_psn=True)
    _test_conv2d_equality(net1, net2, C, T, with_bn=True, with_psn=True)


def test_conv2d_bn_PSN_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dBNPSN(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                bn=nn.BatchNorm2d(C),
                neuron=PSN(T=T),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_PSN_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}bn", nn.BatchNorm2d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", PSN(T=T))
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_SlidingPSN_equality():
    T = 4
    C = 48
    net1 = Conv2dBNSlidingPSN(
        proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
        bn=nn.BatchNorm2d(C),
        neuron=SlidingPSN(k=T // 2),
        spike_compressor=IdentitySpikeCompressor()
    )
    net2 = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net1, net2, C, T, with_bn=True, with_psn=True)
    _test_conv2d_equality(net1, net2, C, T, with_bn=True, with_psn=True)


def test_conv2d_bn_SlidingPSN_memory_blocked():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(
            f"{i}",
            Conv2dBNSlidingPSN(
                proj=nn.Conv2d(C, C, 3, padding=1, bias=True),
                bn=nn.BatchNorm2d(C),
                neuron=SlidingPSN(k=T // 2),
                spike_compressor=IdentitySpikeCompressor()
            ),
        )
    _test_conv2d_memory(net, C, T)


def test_conv2d_bn_SlidingPSN_memory_conventional():
    T = 4
    C = 32
    net = nn.Sequential()
    for i in range(50):
        net.add_module(f"{i}merge", MergeTN())
        net.add_module(f"{i}proj", nn.Conv2d(C, C, 3, padding=1, bias=True))
        net.add_module(f"{i}bn", nn.BatchNorm2d(C))
        net.add_module(f"{i}split", SplitTN(T))
        net.add_module(f"{i}neuron", SlidingPSN(k=T // 2))
    _test_conv2d_memory(net, C, T)


if __name__ == "__main__":
    print("=" * 20, "LinearLIF", "=" * 20)
    test_linear_lif_equality()
    test_linear_lif_memory_blocked()
    test_linear_lif_memory_conventional()

    print("=" * 20, "LinearBNLIF", "=" * 20)
    test_linear_bn_lif_equality()
    test_linear_bn_lif_memory_blocked()
    test_linear_bn_lif_memory_conventional()

    print("=" * 20, "LinearPSN", "=" * 20)
    test_linear_PSN_equality()
    test_linear_PSN_memory_blocked()
    test_linear_PSN_memory_conventional()

    print("=" * 20, "LinearBNPSN", "=" * 20)
    test_linear_bn_PSN_equality()
    test_linear_bn_PSN_memory_blocked()
    test_linear_bn_PSN_memory_conventional()

    print("=" * 20, "LinearSlidingPSN", "=" * 20)
    test_linear_SlidingPSN_equality()
    test_linear_SlidingPSN_memory_blocked()
    test_linear_SlidingPSN_memory_conventional()

    print("=" * 20, "LinearBNSlidingPSN", "=" * 20)
    test_linear_bn_SlidingPSN_equality()
    test_linear_bn_SlidingPSN_memory_blocked()
    test_linear_bn_SlidingPSN_memory_conventional()

    print("=" * 20, "Conv2dLIF", "=" * 20)
    test_conv2d_lif_equality()
    test_conv2d_lif_memory_blocked()
    test_conv2d_lif_memory_conventional()

    print("=" * 20, "Conv2dBNLIF", "=" * 20)
    test_conv2d_bn_lif_equality()
    test_conv2d_bn_lif_memory_blocked()
    test_conv2d_bn_lif_memory_conventional()

    print("=" * 20, "Conv2dPSN", "=" * 20)
    test_conv2d_PSN_equality()
    test_conv2d_PSN_memory_blocked()
    test_conv2d_PSN_memory_conventional()

    print("=" * 20, "Conv2dBNPSN", "=" * 20)
    test_conv2d_bn_PSN_equality()
    test_conv2d_bn_PSN_memory_blocked()
    test_conv2d_bn_PSN_memory_conventional()

    print("=" * 20, "Conv2dSlidingPSN", "=" * 20)
    test_conv2d_SlidingPSN_equality()
    test_conv2d_SlidingPSN_memory_blocked()
    test_conv2d_SlidingPSN_memory_conventional()

    print("=" * 20, "Conv2dBNSlidingPSN", "=" * 20)
    test_conv2d_bn_SlidingPSN_equality()
    test_conv2d_bn_SlidingPSN_memory_blocked()
    test_conv2d_bn_SlidingPSN_memory_conventional()
