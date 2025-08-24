import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn

from modules import *
from shd.models import *

DEVICE = "cpu"


def make_parameters_equal(net, reference_net):
    """Assume that `net` and `reference_net` have the parameter order.
    """
    for p, ref_p in zip(net.parameters(), reference_net.parameters()):
        p.data = ref_p.data.clone()


def check_equal(x1, x2):
    print(x1.mean(), x2.mean())
    assert (
        torch.allclose(x1, x2, atol=1e-5) or
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0).item() > 0.99
    ), (
        (torch.abs(x1 - x2) / (torch.abs(x1) + 1e-8)).mean(),
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0),
    )


def test_ssa_equality():
    T, N, C, H, W = 10, 8, 16, 8, 8
    net1 = SSACoreLIF(
        scale=0.125,
        neuron=AutogradLIF(),
        spike_compressor=BitSpikeCompressor(),
    )
    net2 = TCSSACoreLIF(
        scale=0.125,
        neuron=AutogradLIF(),
        spike_compressor=BitSpikeCompressor(),
    )
    qkv = torch.randn(3, T, N, C, H, W) + 0.6
    qkv = (qkv >= 0.0).float()
    qkv = qkv.to(DEVICE)

    net1 = net1.to(DEVICE)
    net2 = net2.to(DEVICE)
    make_parameters_equal(net1, net2)

    x1 = qkv.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    loss1 = (s1 - 0.9).pow(2).sum()
    loss1.backward()

    x2 = qkv.clone()
    x2.requires_grad = True
    s2 = net2(x2)
    loss2 = (s2 - 0.9).pow(2).sum()
    loss2.backward()

    print("Firing rate: ", s1.mean().item(), s2.mean().item())
    check_equal(s1, s2)

    check_equal(x1.grad, x2.grad)


def test_linear_lif_equality():
    T, N, L = 32, 16, 70
    net1 = GCContainer(BitSpikeCompressor(), nn.Linear(L, L), AutogradLIF())
    net2 = TCLinearLIF(
        proj=nn.Linear(L, L),
        neuron=AutogradLIF(),
        spike_compressor=BitSpikeCompressor()
    )
    x = torch.randn(T, N, L)
    x = (x >= 0.0).float()
    x = x.to(DEVICE)

    net1 = net1.to(DEVICE)
    net2 = net2.to(DEVICE)
    make_parameters_equal(net1, net2)

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

    print("Firing rate: ", s1.mean().item(), s2.mean().item())
    check_equal(s1, s2)

    check_equal(x1.grad, x2.grad)
    check_equal(net1.proj.weight.grad, net2.proj.weight.grad)


def test_linear_plif_equality():
    T, N, L = 32, 8, 700
    net1 = LinearPLIF(L, 20)
    net2 = LinearPLIFTCCheckpointing(
        L, 20, spike_compressor="BitSpikeCompressor"
    )
    x = torch.rand(T, N, L)
    x = (x >= 0.0).float()
    x = x.to(DEVICE)

    net1 = net1.to(DEVICE)
    net2 = net2.to(DEVICE)
    make_parameters_equal(net1, net2)

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

    print("Firing rate: ", s1.mean().item(), s2.mean().item())
    check_equal(s1, s2)

    check_equal(x1.grad, x2.grad)
    check_equal(net1.dense.weight.grad, net2.dense.weight.grad)
    check_equal(net1._beta.grad, net2._beta.grad)


def test_ssa_memory_gc():
    T, N, C, H, W = 100, 8, 16, 32, 32
    net1 = nn.Sequential(
        SSACoreLIF(
            scale=0.125,
            neuron=AutogradLIF(),
            spike_compressor=NullSpikeCompressor(),
        ),
        MergeTN(),
        nn.AdaptiveAvgPool1d(1),
    )
    net1 = net1.to(DEVICE)

    qkv = torch.randn(3, T, N, C, H, W) + 0.6
    qkv = (qkv >= 0.0).float()
    qkv = qkv.to(DEVICE)
    qkv.requires_grad = True

    torch.cuda.reset_peak_memory_stats(DEVICE)
    y = net1(qkv)
    loss = y.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"GC, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_ssa_memory_tcgc():
    T, N, C, H, W = 100, 8, 16, 32, 32
    net2 = nn.Sequential(
        TCSSACoreLIF(
            scale=0.125,
            neuron=AutogradLIF(),
            spike_compressor=NullSpikeCompressor(),
            n_chunk=4,
        ),
        MergeTN(),
        nn.AdaptiveAvgPool1d(1),
    )
    net2 = net2.to(DEVICE)

    qkv = torch.randn(3, T, N, C, H, W) + 0.6
    qkv = (qkv >= 0.0).float()
    qkv = qkv.to(DEVICE)
    qkv.requires_grad = True

    torch.cuda.reset_peak_memory_stats(DEVICE)
    y = net2(qkv)
    loss = y.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"TCGC, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


if __name__ == "__main__":
    test_linear_lif_equality()
    test_linear_plif_equality()
    # test_ssa_equality()
    # test_ssa_memory_gc()
    # test_ssa_memory_tcgc()
