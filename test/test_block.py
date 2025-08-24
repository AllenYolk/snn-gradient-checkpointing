import copy
import sys

sys.path.insert(0, "./src")

import torch
import torch.nn as nn

from modules import *

DEVICE = "cpu"


def _check_equal(x1, x2):
    assert (
        torch.allclose(x1, x2, atol=1e-5) or
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0).item() > 0.99
    ), (
        (torch.abs(x1 - x2) / (torch.abs(x1) + 1e-8)).mean(),
        F.cosine_similarity(x1.flatten(), x2.flatten(), dim=0),
    )


def _test_linear_equatlity(net, C=128, T=100, with_bn=False, with_psn=False):
    N = 64
    x = torch.randn(T, N, C) + 0.6
    x = (x >= 0.0).float()
    x = x.to(DEVICE)

    net1 = net.to(DEVICE)
    net2 = copy.deepcopy(net)
    net2_ = GCContainer(IdentitySpikeCompressor(), *net2)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    loss1 = (s1 - 0.9).pow(2).sum()
    loss1.backward()

    x2 = x.clone()
    x2.requires_grad = True
    s2 = net2_(x2)
    loss2 = (s2 - 0.9).pow(2).sum()
    loss2.backward()

    _check_equal(s1, s2)
    if with_psn:
        idx = 4 if with_bn else 1
        _check_equal(net1[idx].weight.grad, net2[idx].weight.grad)
        _check_equal(net1[idx].bias.grad, net2[idx].bias.grad)
    _check_equal(x1.grad, x2.grad)
    if with_bn:
        _check_equal(net1[2].running_mean, net2[2].running_mean)
        _check_equal(net1[2].running_var, net2[2].running_var)
        _check_equal(net1[2].weight.grad, net2[2].weight.grad)
        _check_equal(net1[2].bias.grad, net2[2].bias.grad)
    _check_equal(net1[0].weight.grad, net2[0].weight.grad)
    _check_equal(net1[0].bias.grad, net2[0].bias.grad)
    print("Equal firing rate! Firing rate: ", s1.mean().item())


def _test_conv2d_equality(net, C=48, T=20, with_bn=False, with_psn=False):
    N = 64
    H = 32
    W = 32
    x = torch.randn(T, N, C, H, W) + 0.6
    x = (x >= 0.0).float()
    x = x.to(DEVICE)

    net1 = net.to(DEVICE)
    net2 = copy.deepcopy(net).to(DEVICE)
    f_net2 = checkpointing.to_gc_function(IdentitySpikeCompressor(), net2)

    x1 = x.clone()
    x1.requires_grad = True
    s1 = net1(x1)
    loss1 = (s1 - 0.9).pow(2).sum()
    loss1.backward()

    x2 = x.clone()
    x2.requires_grad = True
    s2 = f_net2(x2)
    loss2 = (s2 - 0.9).pow(2).sum()
    loss2.backward()

    _check_equal(s1, s2)
    _check_equal(x1.grad, x2.grad)
    _check_equal(net1[1].weight.grad, net2[1].weight.grad)
    _check_equal(net1[1].bias.grad, net2[1].bias.grad)
    if with_bn:
        _check_equal(net1[2].running_mean, net2[2].running_mean)
        _check_equal(net1[2].running_var, net2[2].running_var)
        _check_equal(net1[2].weight.grad, net2[2].weight.grad)
        _check_equal(net1[2].bias.grad, net2[2].bias.grad)
    if with_psn:
        idx = 4 if with_bn else 3
        _check_equal(net1[idx].weight.grad, net2[idx].weight.grad)
        _check_equal(net1[idx].bias.grad, net2[idx].bias.grad)
    print("Equal firing rate! Firing rate: ", s1.mean().item())


def test_linear_lif_equality():
    C = 128
    T = 100
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SJLIF(decay_lambda=0.5, backend="torch"),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False)

    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SJLIF(decay_lambda=0.5, backend="torch"),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False)


def test_linear_PSN_equality():
    C = 128
    T = 100
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        PSN(T=T),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False, with_psn=True)
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        PSN(T=T),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False, with_psn=True)


def test_linear_SlidingPSN_equality():
    C = 128
    T = 100
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SlidingPSN(k=T // 5),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False, with_psn=True)
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        SlidingPSN(k=T // 5),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=False, with_psn=True)


def test_linear_bn_lif_equality():
    T = 100
    C = 128

    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=True)
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    ).to(DEVICE)
    _test_linear_equatlity(net, C, T, with_bn=True)


def test_linear_bn_PSN_equality():
    C = 128
    T = 100
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_linear_equatlity(net, C, T, with_bn=True, with_psn=True)
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_linear_equatlity(net, C, T, with_bn=True, with_psn=True)


def test_linear_bn_SlidingPSN_equality():
    C = 128
    T = 100
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SlidingPSN(k=T // 5),
    )
    _test_linear_equatlity(net, C, T, with_bn=True, with_psn=True)
    net = nn.Sequential(
        nn.Linear(C, C, bias=True),
        MergeTN(),
        nn.BatchNorm1d(C),
        SplitTN(T),
        SlidingPSN(k=T // 5),
    )
    _test_linear_equatlity(net, C, T, with_bn=True, with_psn=True)


def test_conv2d_lif_equality():
    C = 48
    T = 20
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net, C, T, with_bn=False)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net, C, T, with_bn=False)


def test_conv2d_PSN_equality():
    T = 20
    C = 48
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net, C, T, with_bn=False, with_psn=True)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net, C, T, with_bn=False, with_psn=True)


def test_conv2d_SlidingPSN_equality():
    T = 4
    C = 48
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net, C, T, with_bn=False, with_psn=True)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net, C, T, with_bn=False, with_psn=True)


def test_conv2d_bn_lif_equality():
    T = 20
    C = 48
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net, C, T, with_bn=True)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SJLIF(decay_lambda=0.5, backend="torch"),
    )
    _test_conv2d_equality(net, C, T, with_bn=True)


def test_conv2d_bn_PSN_equality():
    T = 4
    C = 15
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net, C, T, with_bn=True, with_psn=True)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        PSN(T=T),
    )
    _test_conv2d_equality(net, C, T, with_bn=True, with_psn=True)


def test_conv2d_bn_SlidingPSN_equality():
    T = 4
    C = 48
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net, C, T, with_bn=True, with_psn=True)
    net = nn.Sequential(
        MergeTN(),
        nn.Conv2d(C, C, 3, padding=1, bias=True),
        nn.BatchNorm2d(C),
        SplitTN(T),
        SlidingPSN(k=T // 2),
    )
    _test_conv2d_equality(net, C, T, with_bn=True, with_psn=True)


if __name__ == "__main__":
    print("=" * 20, "LinearLIF", "=" * 20)
    test_linear_lif_equality()

    print("=" * 20, "LinearBNLIF", "=" * 20)
    test_linear_bn_lif_equality()

    print("=" * 20, "LinearPSN", "=" * 20)
    test_linear_PSN_equality()

    print("=" * 20, "LinearBNPSN", "=" * 20)
    test_linear_bn_PSN_equality()

    print("=" * 20, "LinearSlidingPSN", "=" * 20)
    test_linear_SlidingPSN_equality()

    print("=" * 20, "LinearBNSlidingPSN", "=" * 20)
    test_linear_bn_SlidingPSN_equality()

    print("=" * 20, "Conv2dLIF", "=" * 20)
    test_conv2d_lif_equality()

    print("=" * 20, "Conv2dBNLIF", "=" * 20)
    test_conv2d_bn_lif_equality()

    print("=" * 20, "Conv2dPSN", "=" * 20)
    test_conv2d_PSN_equality()

    print("=" * 20, "Conv2dBNPSN", "=" * 20)
    test_conv2d_bn_PSN_equality()

    print("=" * 20, "Conv2dSlidingPSN", "=" * 20)
    test_conv2d_SlidingPSN_equality()

    print("=" * 20, "Conv2dBNSlidingPSN", "=" * 20)
    test_conv2d_bn_SlidingPSN_equality()
