import sys

sys.path.insert(0, "./src")

import torch

from models import SJLIFNode, HandWrittenLIFNode, MELIFNode


def test_vanilla_lif():
    T = 1000
    N = 100
    C = 200
    x = torch.randn(T, N, C) + 0.6

    net = SJLIFNode()
    x.requires_grad = True
    net = net.to("cuda:0")
    x = x.to("cuda:0")

    torch.cuda.reset_peak_memory_stats("cuda:0")
    s = net(x)
    loss = s.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"Vanilla LIF, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_handwritten_lif():
    T = 1000
    N = 100
    C = 200
    x = torch.randn(T, N, C) + 0.6

    net = HandWrittenLIFNode()
    x.requires_grad = True
    net = net.to("cuda:0")
    x = x.to("cuda:0")

    torch.cuda.reset_peak_memory_stats("cuda:0")
    s = net(x)
    loss = s.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"HandWrittenLIF, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


def test_me_lif():
    T = 1000
    N = 100
    C = 200
    x = torch.randn(T, N, C) + 0.6

    net = MELIFNode()
    x.requires_grad = True
    net = net.to("cuda:0")
    x = x.to("cuda:0")

    torch.cuda.reset_peak_memory_stats("cuda:0")
    s = net(x)
    loss = s.sum()
    loss.backward()
    mem_stats = torch.cuda.memory_stats("cuda:0")
    peak_allocated = mem_stats["allocated_bytes.all.peak"] / (1024**2)
    peak_reserved = mem_stats["reserved_bytes.all.peak"] / (1024**2)
    print(
        f"MELIF, Peak allocated: {peak_allocated:.2f} MB, "
        f"Peak reserved: {peak_reserved:.2f} MB"
    )


if __name__ == "__main__":
    test_vanilla_lif()
    test_handwritten_lif()
    test_me_lif()
